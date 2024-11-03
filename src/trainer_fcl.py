import copy
import datetime
import json
import os
import pickle
import random
import time
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from Datasets.data_manager import DataManager
from Datasets.data_split import SplitDataset
from Methods import get_fed_method
from Models import get_model
from sklearn.metrics import confusion_matrix
from utils.conf import log_path


def train(args):
    log_dir = os.path.join(
        log_path(),
        "async" if args.async_task else "sync",
        args.dataset,
        args.model,
        args.exp,
    )
    seed_list = args.seeds.split(",")

    for seed in seed_list:
        args.seed = int(seed)
        args.log_dir = f"{log_dir}/seed{args.seed}"
        if "pmae" in args.method:
            args.images_dir = f"{args.log_dir}/gen_images"

        print(f"seed: {args.seed}")
        print(f"log_dir: {args.log_dir}")
        if "pmae" in args.method:
            print(f"images_dir: {args.images_dir}")

        _train(args)


def _train(args):
    set_random(args.seed)

    sdata = SplitDataset(args)
    datamanager_global = DataManager(
        -1,
        sdata.idata,
        args.init_class,
        args.inc_class,
        shuffle=args.shuffle,
        is_global=True,
    )
    args.n_class = sdata.idata.n_class
    args.n_task = datamanager_global.n_task
    args.class_order = datamanager_global.class_order
    if hasattr(args, "memory_size"):
        args.memory_size_client = args.memory_size // args.client_num

    set_exp(args)

    network_global = get_model(None, -1, args).network
    fed_method = get_fed_method(args)
    fed_method.ini(network_global)

    if hasattr(args, "alpha"):
        idatas = sdata.partition_quantity_skew_datasets()
    else:
        idatas = sdata.partition_label_skew_datasets()
    visualization(sdata.net_cls_counts, args)

    datamanagers_list = [
        DataManager(
            i,
            idatas[i],
            args.init_class,
            args.inc_class,
            shuffle=args.shuffle,
            async_task=args.async_task,
            class_order=args.class_order,
        )
        for i in range(args.client_num)
    ]
    clients_list = [
        get_model(network_global, client_id, args)
        for client_id in range(args.client_num)
    ]

    start_time = time.time()

    fed_method.global_net.to(fed_method.device)
    w_global = copy.deepcopy(fed_method.global_net.state_dict())
    communication_epoch = args.communication_epoch
    for global_epoch in range(communication_epoch):
        fed_method.global_epoch = global_epoch
        local_task = (global_epoch * datamanager_global.n_task) // communication_epoch
        print(f"Global epoch: {global_epoch}")

        total_clients = list(range(len(clients_list)))
        online_clients_list = random.sample(total_clients, k=fed_method.online_num)
        print("Selected client id:", online_clients_list)

        fed_method.check_online_clients_list(
            local_task, clients_list, datamanagers_list, online_clients_list
        )

        # Client
        w_local_list = fed_method.local_update(
            w_global, local_task, clients_list, datamanagers_list, global_epoch
        )

        # Server
        w_global = fed_method.server_update(
            w_local_list, local_task, clients_list, datamanager_global, global_epoch
        )

        eval_model(args, fed_method, global_epoch, local_task, datamanager_global)

    end_time = time.time()
    cost_time = end_time - start_time
    cost_time_str = str(datetime.timedelta(seconds=int(cost_time)))
    print(f"Cost time: {cost_time_str}\n")


def eval_model(args, fed_method, global_epoch, local_task, datamanager_global):
    class_acc, targets, predicts = fed_method.compute_accuracy(
        local_task, datamanager_global
    )
    results = {
        "global_epoch": global_epoch,
        "local_task": local_task,
        "class_acc": class_acc,
    }
    save_results(f"accs_test", results, args.log_dir)
    cmt = confusion_matrix(targets, predicts)
    cmt_path = os.path.join(args.log_dir, "cmts", f"epoch_{global_epoch}_cmt.npy")
    np.save(cmt_path, cmt)
    print(f"Global epoch: {global_epoch}, Local task: {local_task}, Acc: {class_acc}")


def set_exp(args):
    class ConfigEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, type):
                return {"$class": o.__module__ + "." + o.__name__}
            elif isinstance(o, Enum):
                return {
                    "$enum": o.__module__ + "." + o.__class__.__name__ + "." + o.name
                }
            elif callable(o):
                return {"$function": o.__module__ + "." + o.__name__}
            return json.JSONEncoder.default(self, o)

    dirs = [args.log_dir]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        os.makedirs(f"{dir}/cmts", exist_ok=True)
        if "pmae" in args.method:
            os.makedirs(args.images_dir, exist_ok=True)
        config_filepath = os.path.join(dir, "configs.json")
        with open(config_filepath, "w") as fd:
            json.dump(vars(args), fd, indent=2, sort_keys=True, cls=ConfigEncoder)
    result_types = ["test"]
    for result_type in result_types:
        results_filepath = os.path.join(args.log_dir, f"accs_{result_type}.csv")
        with open(results_filepath, "w", encoding="utf-8") as f:
            if os.path.getsize(results_filepath) != 0:
                f.truncate(0)


def save_results(save_name, results, result_dir):
    file_path = os.path.join(result_dir, f"{save_name}.csv")
    with open(file_path, mode="a", encoding="utf-8") as f:
        if os.path.getsize(file_path) == 0:
            for key, value in results.items():
                f.write(f"{key},")
            f.write(f"\n")
        for key, value in results.items():
            f.write(f"{value},")
        f.write(f"\n")


def visualization(clients_data, args):
    file_path = os.path.join(args.log_dir, "clients_data.pkl")
    with open(file_path, "wb") as file:
        pickle.dump(clients_data, file)

    num_clients = args.client_num

    plt.subplots()
    colors = [key for key in matplotlib.colors.CSS4_COLORS.keys()]
    random.shuffle(colors)

    max_offset = 0
    client_height = 1
    for cid in range(num_clients):
        data = clients_data[cid]
        offset = 0
        y_bottom = cid - client_height / 2.0
        y_top = cid + client_height / 2.0

        for lbi, value in data.items():
            plt.fill_between(
                [offset, offset + value],
                y_bottom,
                y_top,
                facecolor=colors[lbi % len(colors)],
            )
            offset += value

        max_offset = max(offset, max_offset)

    print("Max volume", max_offset)
    plt.xlim(0, max_offset)
    plt.ylim(-0.5, num_clients - 0.5)
    plt.ylabel("Client ID")
    plt.xlabel("Number of Samples")
    plt.savefig(os.path.join(args.log_dir, "visualization.png"))
    plt.show()


def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
