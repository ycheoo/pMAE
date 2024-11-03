import copy

import numpy as np
import torch.nn as nn
from utils.conf import get_device


class LocalMethod(nn.Module):
    """
    Federated learning Methods.
    """

    def __init__(self, args) -> None:
        super(LocalMethod, self).__init__()

        self.args = args
        self.device = get_device(device_id=self.args.device_id)

    def local_check(self, task, clients_list, datamanagers_list, online_clients_list):
        # update state for alignment
        for client_id in range(len(clients_list)):
            clients_list[client_id].update_state(task, datamanagers_list[client_id])

        checked_online_clients_list = copy.deepcopy(online_clients_list)
        for i, client_id in enumerate(online_clients_list):
            checking_result = "OK"
            if not clients_list[client_id].check_loader(
                task, datamanagers_list[client_id]
            ):
                checking_result = (
                    "No current task data, delete from online clients list"
                )
                checked_online_clients_list.remove(client_id)
            print(
                f"Client {client_id} local checking... => {checking_result} [{i+1}/{len(online_clients_list)}]"
            )
        return checked_online_clients_list

    def loc_update(self):
        pass

    def train_net(self):
        pass
