import copy

import numpy as np
import torch
from Backbones.inc_net import PromptVitNet
from Datasets.data_manager import DataManager
from Models.meta_model import BaseLearner
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.toolkit import tensor2numpy


class Learner(BaseLearner):
    def __init__(self, network_global, client_id, args):
        super().__init__(client_id, args)
        if network_global is None:
            self.network = PromptVitNet(args)
        else:
            self.network = network_global

        if self.client_id != -1:
            print(f"Client {self.client_id} loaded model")
        else:
            g_prompt_params = self.network.backbone.g_prompt.numel()
            e_prompt_params = sum(
                p.numel() for p in self.network.backbone.e_prompt.parameters()
            )
            fc_params = sum(p.numel() for p in self.network.backbone.head.parameters())
            print(g_prompt_params, e_prompt_params, fc_params)

    def train(self, w_global, train_loader):
        print(f"Learning on: {self.cur_classes} (Global task {self.cur_task_global})")

        self.network.load_state_dict(w_global)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self.cur_task > 0:
            if self.cur_task_global not in self.known_tasks:
                print("Prompt transferred from previous task")
                self.init_prompt(optimizer)
            elif self.new_task:
                print("Prompt transferred from other clients")

        if self.cur_task > 0 and self.args.reinit_optimizer:
            optimizer = self.get_optimizer()

        w_local = self.init_train(train_loader, optimizer, scheduler)
        return w_local

    def get_optimizer(self):
        if self.args.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.network.backbone.parameters()),
                lr=self.init_lr,
                weight_decay=self.weight_decay,
            )
        return optimizer

    def get_scheduler(self, optimizer):
        if self.args.scheduler == "constant":
            scheduler = None

        return scheduler

    def init_prompt(self, optimizer):
        args = self.args
        model = self.network.backbone
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            prev_start = self.pre_task_global * args.top_k
            prev_end = (self.pre_task_global + 1) * args.top_k

            cur_start = self.cur_task_global
            cur_end = (self.cur_task_global + 1) * args.top_k

            if (prev_end > args.size) or (cur_end > args.size):
                pass
            else:
                cur_idx = (
                    (slice(None), slice(None), slice(cur_start, cur_end))
                    if args.use_prefix_tune_for_e_prompt
                    else (slice(None), slice(cur_start, cur_end))
                )
                prev_idx = (
                    (slice(None), slice(None), slice(prev_start, prev_end))
                    if args.use_prefix_tune_for_e_prompt
                    else (slice(None), slice(prev_start, prev_end))
                )

                with torch.no_grad():
                    model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                    optimizer.param_groups[0]["params"] = model.parameters()

    def init_train(self, train_loader, optimizer, scheduler):
        info = "NO INFO"
        prog_bar = tqdm(range(self.tuned_epoch))
        self.network.backbone.train()
        self.network.original_backbone.eval()
        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.network(inputs, task_id=self.cur_task_global, train=True)
                logits = output["logits"]
                class_mask = [
                    col_index
                    for col_index in range(logits.shape[1])
                    if col_index not in self.cur_classes
                ]
                logits[:, class_mask] = float("-inf")

                loss = F.cross_entropy(logits, targets.long())
                if self.args.pull_constraint and "reduce_sim" in output:
                    loss = loss - self.args.pull_constraint_coeff * output["reduce_sim"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self.cur_task,
                epoch + 1,
                self.tuned_epoch,
                losses / len(train_loader),
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
