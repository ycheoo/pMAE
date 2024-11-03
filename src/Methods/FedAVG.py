from Methods.meta_method import FederatedMethod


class Method(FederatedMethod):
    def __init__(self, args):
        super().__init__(args)

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
            datamanager_global=datamanager_global,
            local_task=local_task,
            w_local_list=w_local_list,
        )
        return w_global
