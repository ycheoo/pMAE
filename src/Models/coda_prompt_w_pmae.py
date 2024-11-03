import copy
import math

import numpy as np
import torch
from Backbones.inc_net import CodaPromptVitNet
from Datasets.data_manager import DataManager
from Models.meta_model import BaseLearner
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
            self.network = CodaPromptVitNet(args)
        else:
            self.network = network_global

        if self.client_id != -1:
            print(f"Client {self.client_id} loaded model")
        else:
            prompt_cls_params = sum(p.numel() for p in self.network.prompt.parameters())
            prompt_mae_params = self.network.pmae.prompt_mae.numel()
            fc_params = sum(p.numel() for p in self.network.fc.parameters())
            print(prompt_cls_params, prompt_mae_params, fc_params)

    def train(self, w_global, train_loader, data_manager: DataManager):
        print(f"Learning on: {self.cur_classes} (Global task {self.cur_task_global})")

        self.network.load_state_dict(w_global)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self.new_task:
            self.network.prompt.set_prompt(self.cur_task_global, self.known_tasks)
            if self.cur_task_global not in self.known_tasks:
                print("Prompt schmidt init")
                self.network.prompt.process_gram_schmidt()
            else:
                print("Prompt transferred from other clients")

        if self.cur_task > 0 and self.args.reinit_optimizer:
            optimizer = self.get_optimizer()

        w_local = self.init_train(train_loader, optimizer, scheduler)

        agg_input, agg_latent, agg_mask, agg_ids_restore = self.extract_tokens(
            data_manager
        )

        return w_local, agg_input, agg_latent, agg_mask, agg_ids_restore

    def get_optimizer(self):
        cls_params = {
            "params": list(self.network.prompt.parameters())
            + list(self.network.fc.parameters()),
            "lr": self.init_lr,
            "weight_decay": self.weight_decay,
        }
        mae_params = {
            "params": [self.network.pmae.prompt_mae],
            "lr": self.init_lr,
            "weight_decay": self.weight_decay,
        }
        params = [cls_params, mae_params]
        if self.args.optimizer == "adam":
            optimizer = optim.Adam(
                params, lr=self.init_lr, weight_decay=self.weight_decay
            )
        return optimizer

    def get_scheduler(self, optimizer):
        if self.args.scheduler == "cosine":
            scheduler = CosineSchedule(optimizer, K=self.tuned_epoch)
        elif self.args.scheduler == "constant":
            scheduler = None

        return scheduler

    def init_train(self, train_loader, optimizer, scheduler):
        info = "NO INFO"
        prog_bar = tqdm(range(self.tuned_epoch))
        for _, epoch in enumerate(prog_bar):
            self.network.train()

            losses_CE, losses_MSE = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits, prompt_loss = self.network(inputs, train=True)
                class_mask = [
                    col_index
                    for col_index in range(logits.shape[1])
                    if col_index not in self.cur_classes
                ]
                logits[:, class_mask] = float("-inf")

                loss_supervised = F.cross_entropy(logits, targets.long())
                loss_CE = loss_supervised + prompt_loss.sum()

                inputs = self.network.pmae.normalize(inputs)
                loss_MSE, pred, mask = self.network.pmae.forward_mae(inputs)

                loss = loss_CE + self.lamda * loss_MSE

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses_CE += loss_CE.item()
                losses_MSE += loss_MSE.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
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
            inputs = self.network.pmae.normalize(inputs)
            with torch.no_grad():
                latent, mask, ids_restore = self.network.pmae.forward_encoder(inputs)
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


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineSchedule(_LRScheduler):
    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos(
            (99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1))
        )

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]
