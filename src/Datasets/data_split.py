import random

import numpy as np
from Datasets.data import iData, iCUB, iImageNetR


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    y_train = np.array(y_train)
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print("mean:", np.mean(data_list))
    print("std:", np.std(data_list))
    # print("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts


class SplitDataset(object):
    def __init__(self, args) -> None:
        self.alpha = getattr(args, "alpha", None)
        self.beta = getattr(args, "beta", None)
        self.dataset = args.dataset
        self.client_num = args.client_num

        self.idata = self.get_idata()
        self.idata.download_data()

    def get_idata(self) -> iData:
        if "imagenetr" in self.dataset:
            return iImageNetR()
        elif "cub" in self.dataset:
            return iCUB()
        else:
            raise NotImplementedError("Unknown dataset {}.".format(self.dataset))

    def partition_quantity_skew_datasets(self):
        n_participants = self.client_num
        x_train = self.idata.train_data
        y_train = self.idata.train_targets
        net_dataidx_map = {}
        K = self.idata.n_class
        C = self.alpha * self.idata.n_class
        times = [0 for i in range(K)]
        contain = []
        for i in range(n_participants):
            current = [i % K]
            times[i % K] += 1
            j = 1
            while j < C:
                ind = random.randint(0, K - 1)
                if ind not in current:
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        net_dataidx_map = {
            i: np.ndarray(0, dtype=np.int64) for i in range(n_participants)
        }
        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(n_participants):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1
        self.net_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

        idatas = [self.get_idata() for j in range(n_participants)]
        for j in range(n_participants):
            idatas[j].load_data(
                x_train[net_dataidx_map[j]],
                y_train[net_dataidx_map[j]],
                self.idata.test_data,
                self.idata.test_targets,
            )
        return idatas

    def partition_label_skew_datasets(self):
        n_class = self.idata.n_class
        n_participants = self.client_num
        min_size = 0
        min_require_size = 10
        x_train = self.idata.train_data
        y_train = self.idata.train_targets
        N = len(y_train)
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_participants)]
            for k in range(n_class):
                idx_k = [i for i, j in enumerate(y_train) if j == k]
                np.random.shuffle(idx_k)

                beta = self.beta
                if beta == 0:  # iid
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            idx_batch, np.array_split(idx_k, n_participants)
                        )
                    ]
                else:  # label-skewed
                    proportions = np.random.dirichlet(
                        np.repeat(a=beta, repeats=n_participants)
                    )
                    proportions = np.array(
                        [
                            p * (len(idx_j) < N / n_participants)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(n_participants):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        self.net_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

        idatas = [self.get_idata() for j in range(n_participants)]
        for j in range(n_participants):
            idatas[j].load_data(
                x_train[net_dataidx_map[j]],
                y_train[net_dataidx_map[j]],
                self.idata.test_data,
                self.idata.test_targets,
            )
        return idatas
