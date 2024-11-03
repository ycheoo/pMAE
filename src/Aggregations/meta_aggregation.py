from abc import abstractmethod
from argparse import Namespace


class FederatedAggregation:
    """
    Federated Aggregation
    """

    NAME = None

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @abstractmethod
    def weight_calculate(self, clients_list, online_clients_list):
        pass

    def agg_parts(self, freq, w_local_list, global_net, except_part):
        global_w = {}

        first = True
        for index, net_para in enumerate(w_local_list):
            used_net_para = {}
            for k, v in net_para.items():
                is_in = False
                for part_str_index in range(len(except_part)):
                    if except_part[part_str_index] in k:
                        is_in = True
                        break

                if not is_in:
                    used_net_para[k] = v

            if first:
                first = False
                for key in used_net_para:
                    global_w[key] = used_net_para[key] * freq[index]
            else:
                for key in used_net_para:
                    global_w[key] += used_net_para[key] * freq[index]

        global_net.load_state_dict(global_w, strict=False)
        print(f"Global model updated")

        return
