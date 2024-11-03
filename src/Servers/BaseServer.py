import copy

import numpy as np
import torch
from Servers.meta_server import SeverMethod
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Server(SeverMethod):
    def __init__(self, args):
        super().__init__(args)

    def sever_update(
        self,
        fed_aggregation,
        online_clients_list,
        global_net,
        clients_list,
        datamanager_global,
        local_task,
        w_local_list,
    ):
        self.global_known_tasks = np.unique(
            self.global_known_tasks
            + [clients_list[i].cur_task_global for i in online_clients_list]
        ).tolist()
        print("Global Tasks:", self.global_known_tasks)
        for client in clients_list:
            client.known_tasks = copy.deepcopy(self.global_known_tasks)

        freq = fed_aggregation.weight_calculate(
            clients_list=clients_list, online_clients_list=online_clients_list
        )

        fed_aggregation.agg_parts(
            freq=freq,
            w_local_list=w_local_list,
            except_part=self.args.freeze if hasattr(self.args, "freeze") else [],
            global_net=global_net,
        )

        w_global = copy.deepcopy(global_net.state_dict())
        
        return w_global
