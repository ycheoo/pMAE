from Methods.meta_method import FederatedMethod


class Method(FederatedMethod):
    def __init__(self, args):
        super().__init__(args)
        self.local_input = {}
        self.local_latent = {}
        self.local_mask = {}
        self.local_ids_restore = {}

    def check_online_clients_list(
        self, task, clients_list, datamanagers_list, online_clients_list
    ):
        self.online_clients_list = online_clients_list
        self.online_clients_list = self.local_model.local_check(
            task=task,
            online_clients_list=self.online_clients_list,
            clients_list=clients_list,
            datamanagers_list=datamanagers_list,
        )

    def local_update(
        self, w_global, task, clients_list, datamanagers_list, global_epoch
    ):
        w_local_list = self.local_model.loc_update(
            w_global=w_global,
            task=task,
            online_clients_list=self.online_clients_list,
            clients_list=clients_list,
            datamanagers_list=datamanagers_list,
            local_input=self.local_input,
            local_latent=self.local_latent,
            local_mask=self.local_mask,
            local_ids_restore=self.local_ids_restore,
        )
        return w_local_list

    def server_update(
        self, w_local_list, local_task, clients_list, datamanager_global, global_epoch
    ):
        w_global = self.server_model.sever_update(
            fed_aggregation=self.fed_aggregation,
            online_clients_list=self.online_clients_list,
            global_net=self.global_net,
            clients_list=clients_list,
            local_input=self.local_input,
            local_latent=self.local_latent,
            local_mask=self.local_mask,
            local_ids_restore=self.local_ids_restore,
            datamanager_global=datamanager_global,
            local_task=local_task,
            global_epoch=global_epoch,
            w_local_list=w_local_list,
        )
        return w_global
