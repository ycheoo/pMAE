import copy

import numpy as np
import torch
from Backbones.inc_net import PromptMAE
from Datasets.data_manager import DataManager
from Models.meta_model import BaseLearner
from PIL import Image
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy


class Learner(BaseLearner):
    def __init__(self, network_global, client_id, args):
        super().__init__(client_id, args)
        self.lamda = args.lamda
        self.num_upload_recon = args.upload_recon
        if network_global is None:
            self.network = PromptMAE(args)
        else:
            self.network = network_global

        if self.client_id != -1:
            print(f"Client {self.client_id} loaded model")
        else:
            prompt_cls_params = self.network.prompt_cls.numel()
            prompt_mae_params = self.network.prompt_mae.numel()
            fc_params = sum(p.numel() for p in self.network.fc.parameters())
            print(
                prompt_cls_params,
                prompt_mae_params,
                fc_params,
            )

    def train(self, w_global, train_loader, data_manager: DataManager):
        print(f"Learning on: {self.cur_classes} (Global task {self.cur_task_global})")
        self.network.load_state_dict(w_global)

        w_local = self.init_train(train_loader)

        agg_input, agg_latent, agg_mask, agg_ids_restore = self.extract_tokens(
            data_manager
        )
        return w_local, agg_input, agg_latent, agg_mask, agg_ids_restore

    def init_train(self, train_loader):
        info = "NO INFO"

        cls_params = {
            "params": [self.network.prompt_cls] + list(self.network.fc.parameters()),
            "lr": self.init_lr,
            "weight_decay": self.weight_decay,
        }
        mae_params = {
            "params": [self.network.prompt_mae],
            "lr": self.init_lr,
            "weight_decay": self.weight_decay,
        }
        params = [cls_params, mae_params]

        optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        prog_bar = tqdm(range(self.tuned_epoch))
        for _, epoch in enumerate(prog_bar):
            self.network.train()

            correct, total = 0, 0
            losses_CE, losses_MSE = 0.0, 0.0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.network(inputs, mask_ratio=0)
                logits = output
                class_mask = [
                    col_index
                    for col_index in range(logits.shape[1])
                    if col_index not in self.cur_classes
                ]
                logits[:, class_mask] = float("-inf")
                loss_CE = F.cross_entropy(logits, targets.long())

                inputs = self.network.normalize(inputs)
                loss_MSE, pred, mask = self.network.forward_mae(inputs)

                loss = loss_CE + self.lamda * loss_MSE

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_CE += loss_CE.item()
                losses_MSE += loss_MSE.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => CE Loss {:.3f}, MSE Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                self.tuned_epoch,
                losses_CE / len(train_loader),
                losses_MSE / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)
        print(info)

        net_para = self.network.state_dict()
        w_local = {
            k: copy.deepcopy(v)
            for k, v in net_para.items()
            if not any(except_key in k for except_key in self.except_part)
        }
        return w_local

    def extract_tokens(self, data_manager):
        input_list = []
        target_list = []
        latent_list = []
        mask_list = []
        ids_restore_list = []
        dataset = data_manager.get_dataset(
            self.cur_classes, source="train", mode="test"
        )
        print(f"Extracting tokens {self.cur_classes} ...")
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        for _, inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = self.network.normalize(inputs)
            with torch.no_grad():
                latent, mask, ids_restore = self.network.forward_encoder(inputs)
            input_list.append(inputs)
            target_list.append(targets)
            latent_list.append(latent)
            mask_list.append(mask)
            ids_restore_list.append(ids_restore)
        input_list = torch.cat(input_list, dim=0)
        target_list = torch.cat(target_list, dim=0)
        latent_list = torch.cat(latent_list, dim=0)
        mask_list = torch.cat(mask_list, dim=0)
        ids_restore_list = torch.cat(ids_restore_list, dim=0)

        agg_input = {}
        agg_latent = {}
        agg_mask = {}
        agg_ids_restore = {}
        for class_index in self.cur_classes:
            data_index = (target_list == class_index).nonzero().squeeze(-1)
            if data_index.shape[0] != 0:
                total_indices = data_index.shape[0]
                random_index = (
                    data_index[torch.randperm(total_indices)[: self.num_upload_recon]]
                    if total_indices >= self.num_upload_recon
                    else data_index
                )
                all_input = input_list[random_index]
                agg_input[class_index] = all_input.cpu().numpy()

                all_latent = latent_list[random_index]
                agg_latent[class_index] = all_latent.cpu().numpy()

                all_mask = mask_list[random_index]
                agg_mask[class_index] = all_mask.cpu().numpy()

                all_ids_restore = ids_restore_list[random_index]
                agg_ids_restore[class_index] = all_ids_restore.cpu().numpy()

        return agg_input, agg_latent, agg_mask, agg_ids_restore
