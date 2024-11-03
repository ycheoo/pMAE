import numpy as np
from Aggregations.meta_aggregation import FederatedAggregation


class Weight(FederatedAggregation):
    NAME = "Weight"

    def __init__(self, args) -> None:
        super().__init__(args)

    def weight_calculate(self, clients_list, online_clients_list):
        online_clients_len = [clients_list[i].curdata_size for i in online_clients_list]
        online_clients_all = np.sum(online_clients_len)
        freq = online_clients_len / online_clients_all
        return freq
