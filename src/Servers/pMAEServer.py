import copy
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from Servers.meta_server import SeverMethod
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from utils.toolkit import tensor2numpy

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Server(SeverMethod):
    def __init__(self, args):
        super().__init__(args)
        self.batch_size = args.batch_size
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.server_tuned_epoch = args.server_tuned_epoch
        self.server_init_lr = args.server_init_lr
        self.server_weight_decay = args.server_weight_decay

        self.global_epoch = -1

        self.task_decoder_prompts = {}

    @property
    def last_global_epoch(self):
        return self.global_epoch == (self.args.communication_epoch - 1)

    @property
    def last_task_epoch(self):
        return (
            (self.global_epoch + 1) * self.args.n_task
        ) % self.args.communication_epoch == 0

    def save_images(self, label, input, mask, images):
        mask = np.transpose(mask, (0, 2, 3, 1))
        input = np.transpose(input, (0, 2, 3, 1))
        images = np.transpose(images, (0, 2, 3, 1))

        images_ori = np.clip(
            (input * imagenet_std + imagenet_mean) * 255, 0, 255
        ).astype(np.uint8)
        images_masked = np.clip(
            ((input * (1 - mask)) * imagenet_std + imagenet_mean) * 255, 0, 255
        ).astype(np.uint8)
        images_recon = np.clip(
            (images * imagenet_std + imagenet_mean) * 255, 0, 255
        ).astype(np.uint8)
        images_paste = np.clip(
            ((input * (1 - mask) + images * mask) * imagenet_std + imagenet_mean) * 255,
            0,
            255,
        ).astype(np.uint8)

        save_dir = f"{self.args.images_dir}/{label}"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(input)):
            image_ori = Image.fromarray(images_ori[i])
            image_ori.save(os.path.join(save_dir, f"image_{i}_ori.jpg"))

            image_masked = Image.fromarray(images_masked[i])
            image_masked.save(os.path.join(save_dir, f"image_{i}_masked.jpg"))

            image_recon = Image.fromarray(images_recon[i])
            image_recon.save(os.path.join(save_dir, f"image_{i}_recon.jpg"))

            image_paste = Image.fromarray(images_paste[i])
            image_paste.save(os.path.join(save_dir, f"image_{i}_paste.jpg"))

    def compute_accuracy(self, loader, test_classes, model):
        correct, total = 0, 0
        test_acc, test_acc_task = 0.0, 0.0
        targets_list, predicts_list = [], []
        prog_bar = tqdm(loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                model_outputs = model(inputs)
                outputs = (
                    model_outputs["logits"]
                    if isinstance(model_outputs, dict)
                    else model_outputs
                )
            class_mask = [
                col_index
                for col_index in range(outputs.shape[1])
                if col_index not in test_classes.tolist()
            ]
            outputs[:, class_mask] = float("-inf")
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
            targets_list.append(targets)
            predicts_list.append(predicts.cpu())

            test_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Test_accy {:.2f}".format(
                test_acc,
            )
            prog_bar.set_description(info)
        print(info)

    def balanced_finetune(self, global_net, dataset, labels):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        if self.args.w_pmae:
            if "l2p" in self.args.model or "dualprompt" in self.args.model:
                params = filter(
                    lambda p: p.requires_grad, global_net.backbone.parameters()
                )
            elif "coda_prompt" in self.args.model:
                params = list(global_net.prompt.parameters()) + list(
                    global_net.fc.parameters()
                )
        else:
            params = [global_net.prompt_cls] + list(global_net.fc.parameters())

        optimizer = optim.Adam(
            params, lr=self.server_init_lr, weight_decay=self.server_weight_decay
        )

        global_net.train()
        prog_bar = tqdm(range(self.server_tuned_epoch))
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = global_net(inputs)
                logits = output
                if "l2p" in self.args.model or "dualprompt" in self.args.model:
                    logits = output["logits"]
                class_mask = [
                    col_index
                    for col_index in range(logits.shape[1])
                    if col_index not in labels
                ]
                logits[:, class_mask] = float("-inf")

                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Tasks {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.global_known_tasks,
                epoch + 1,
                self.server_tuned_epoch,
                losses / len(dataloader),
                train_acc,
            )
            prog_bar.set_description(info)
        print(info)

    def aggregation(
        self, tasks, local_input, local_latent, local_mask, local_ids_restore
    ):
        task_agg_input = {}
        task_agg_latent = {}
        task_agg_mask = {}
        task_agg_ids_restore = {}

        for task in tasks:
            agg_latent = defaultdict(list)
            agg_ids_restore = defaultdict(list)

            if self.last_global_epoch:
                agg_input = defaultdict(list)
                agg_mask = defaultdict(list)

            for key in local_latent.keys():
                if f"task_{task}" not in key:
                    continue
                for label, latent in local_latent[key].items():
                    agg_latent[label].append(latent)
                for label, ids_restore in local_ids_restore[key].items():
                    agg_ids_restore[label].append(ids_restore)
                if self.last_global_epoch:
                    for label, input in local_input[key].items():
                        agg_input[label].append(input)
                    for label, mask in local_mask[key].items():
                        agg_mask[label].append(mask)

            agg_latent = {
                label: np.vstack(tokens) for label, tokens in agg_latent.items()
            }
            agg_ids_restore = {
                label: np.vstack(ids) for label, ids in agg_ids_restore.items()
            }

            task_agg_latent[task] = agg_latent
            task_agg_ids_restore[task] = agg_ids_restore

            if self.last_global_epoch:
                agg_input = {
                    label: np.vstack(tokens) for label, tokens in agg_input.items()
                }
                agg_mask = {label: np.vstack(ids) for label, ids in agg_mask.items()}

                task_agg_input[task] = agg_input
                task_agg_mask[task] = agg_mask

        return task_agg_input, task_agg_latent, task_agg_mask, task_agg_ids_restore

    def gen_image_dataset(
        self,
        tasks,
        global_net_pmae,
        task_agg_input,
        task_agg_latent,
        task_agg_mask,
        task_agg_ids_restore,
    ):
        image_tensors = []
        image_labels = []

        for task in tasks:
            agg_latent = task_agg_latent[task]
            agg_ids_restore = task_agg_ids_restore[task]

            all_latent = []
            all_ids_restore = []

            labels = list(agg_latent.keys())
            labels.sort()
            print(f"Task {task}", labels)

            if self.last_global_epoch:
                agg_input = task_agg_input[task]
                agg_mask = task_agg_mask[task]

                all_input = []
                all_mask = []

            all_labels = []

            global_net_pmae.to(self.device)
            for label in labels:
                latent = torch.from_numpy(agg_latent[label])
                ids_restore = torch.from_numpy(agg_ids_restore[label])
                all_latent.append(latent)
                all_ids_restore.append(ids_restore)
                all_labels.append(
                    torch.full((latent.shape[0],), label, dtype=torch.long)
                )
                if self.last_global_epoch:
                    input = torch.from_numpy(agg_input[label])
                    mask = torch.from_numpy(agg_mask[label])
                    all_input.append(input)
                    all_mask.append(mask)

            with torch.no_grad():
                all_latent = torch.cat(all_latent, dim=0).to(self.device)
                all_ids_restore = torch.cat(all_ids_restore, dim=0).to(self.device)
                task_decoder_prompt = self.task_decoder_prompts[task].to(self.device)

                y = global_net_pmae.forward_decoder(
                    all_latent, all_ids_restore, task_decoder_prompt
                )
                images = global_net_pmae.unpatchify(y).cpu()
                if self.last_global_epoch:
                    all_input = torch.cat(all_input, dim=0)
                    all_mask = torch.cat(all_mask, dim=0)
                    all_mask = all_mask.unsqueeze(-1).repeat(
                        1,
                        1,
                        global_net_pmae.backbone.encoder.patch_embed.patch_size[0] ** 2
                        * 3,
                    )  # (N, H*W, p*p*3)
                    all_mask = all_mask.to(self.device)
                    all_mask = global_net_pmae.unpatchify(
                        all_mask
                    ).cpu()  # 1 is removing, 0 is keeping

            start_idx = 0
            all_labels = torch.cat(all_labels, dim=0)
            for label in labels:
                num_images = (all_labels == label).sum().item()
                label_images = images[start_idx : start_idx + num_images]
                label_labels = all_labels[start_idx : start_idx + num_images]

                if self.last_global_epoch:
                    print(f"Saving {num_images} images from label {label}...")

                    label_input = all_input[start_idx : start_idx + num_images]
                    label_mask = all_mask[start_idx : start_idx + num_images]

                    self.save_images(
                        label,
                        label_input.numpy(),
                        label_mask.numpy(),
                        label_images.numpy(),
                    )

                image_tensors.append(label_images)
                image_labels.append(label_labels)

                start_idx += num_images

        image_tensors = torch.cat(image_tensors, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        dataset = TensorDataset(image_tensors, image_labels)
        labels = list(set(image_labels.numpy()))

        return dataset, labels

    def sever_update(
        self,
        fed_aggregation,
        online_clients_list,
        clients_list,
        global_net,
        local_input,
        local_latent,
        local_mask,
        local_ids_restore,
        datamanager_global,
        local_task,
        global_epoch,
        w_local_list,
    ):
        self.global_epoch = global_epoch

        self.global_known_tasks = np.unique(
            self.global_known_tasks
            + [clients_list[i].cur_task_global for i in online_clients_list]
        ).tolist()
        print("Global Tasks:", self.global_known_tasks)
        for client in clients_list:
            client.known_tasks = copy.deepcopy(self.global_known_tasks)

        test_classes = np.arange(
            0, datamanager_global.get_cumulative_task_size(local_task)
        )

        test_dataset = datamanager_global.get_dataset(
            test_classes, source="test", mode="test"
        )
        loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
        )

        tasks = (
            range(local_task + 1)
            if self.last_task_epoch and not self.args.wo_restore_pool
            else range(local_task, local_task + 1)
        )

        task_agg_input, task_agg_latent, task_agg_mask, task_agg_ids_restore = (
            self.aggregation(
                tasks, local_input, local_latent, local_mask, local_ids_restore
            )
        )

        freq = fed_aggregation.weight_calculate(
            clients_list=clients_list, online_clients_list=online_clients_list
        )

        fed_aggregation.agg_parts(
            freq=freq,
            w_local_list=w_local_list,
            except_part=self.args.freeze if hasattr(self.args, "freeze") else [],
            global_net=global_net,
        )

        global_net_pmae = global_net.pmae if self.args.w_pmae else global_net

        self.task_decoder_prompts[local_task] = (
            global_net_pmae.prompt_mae.data.clone().cpu()
        )

        dataset, labels = self.gen_image_dataset(
            tasks,
            global_net_pmae,
            task_agg_input,
            task_agg_latent,
            task_agg_mask,
            task_agg_ids_restore,
        )

        self.balanced_finetune(
            global_net=global_net,
            dataset=dataset,
            labels=labels,
        )
        
        w_global = copy.deepcopy(global_net.state_dict())
        
        return w_global
