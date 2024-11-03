import numpy as np
import torch.nn as nn
import torch.optim as optim
from Locals.meta_local import LocalMethod
from tqdm import tqdm


class Local(LocalMethod):
    def __init__(self, args):
        super().__init__(args)

    def loc_update(
        self,
        w_global,
        task,
        online_clients_list,
        clients_list,
        datamanagers_list,
        local_input,
        local_latent,
        local_mask,
        local_ids_restore,
    ):
        w_local_list = []
        for i, client_id in enumerate(online_clients_list):
            print(
                f"Client {client_id} local updating... [{i+1}/{len(online_clients_list)}]"
            )
            w_local, agg_input, agg_latent, agg_mask, agg_ids_restore = (
                self.local_train(
                    w_global,
                    task,
                    clients_list[client_id],
                    datamanagers_list[client_id],
                )
            )
            w_local_list.append(w_local)
            key = f"id_{client_id}-task_{task}"
            local_input[key] = agg_input
            local_latent[key] = agg_latent
            local_mask[key] = agg_mask
            local_ids_restore[key] = agg_ids_restore
        return w_local_list

    def local_train(self, w_global, task, client, data_manager):
        train_loader = client.before_task(task, data_manager)
        w_local, agg_input, agg_latent, agg_mask, agg_ids_restore = client.train(
            w_global, train_loader, data_manager
        )
        client.after_task(data_manager)
        return w_local, agg_input, agg_latent, agg_mask, agg_ids_restore
