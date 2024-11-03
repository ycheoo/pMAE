from Locals.meta_local import LocalMethod


class Local(LocalMethod):
    def __init__(self, args):
        super().__init__(args)

    def loc_update(
        self, w_global, task, online_clients_list, clients_list, datamanagers_list
    ):

        w_local_list = []

        for i, client_id in enumerate(online_clients_list):
            print(
                f"Client {client_id} local updating... [{i+1}/{len(online_clients_list)}]"
            )
            w_local = self.local_train(
                w_global, task, clients_list[client_id], datamanagers_list[client_id]
            )
            w_local_list.append(w_local)

        return w_local_list

    def local_train(self, w_global, task, client, data_manager):
        train_loader = client.before_task(task, data_manager)
        w = client.train(w_global, train_loader)
        client.after_task(data_manager)
        return w
