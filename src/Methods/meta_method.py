import numpy as np
import torch
import torch.nn as nn
from Aggregations import get_aggregation
from Datasets.data_manager import DataManager
from Locals import get_local
from scipy.spatial.distance import cdist
from Servers import get_server
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.conf import get_device
from utils.toolkit import tensor2numpy

EPSILON = 1e-8


class FederatedMethod(nn.Module):
    """
    Federated learning Methods.
    """

    def __init__(self, args) -> None:
        super(FederatedMethod, self).__init__()
        self.args = args
        self.nme_classifier = getattr(args, "nme_classifier", False)
        self.online_num = np.ceil(args.client_num * args.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=args.device_id)

        self.local_model = get_local(args)
        self.server_model = get_server(args)

        self.fed_aggregation = get_aggregation(args)

        self.global_epoch = 0

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory

        self.async_task = args.async_task

    def ini(self, network_global):
        self.global_net = network_global
        print(f"Global model loaded")

    def compute_accuracy(self, local_task, data_manager: DataManager):
        if "coda_prompt" in self.args.model:
            self.global_net.prompt.known_tasks = self.server_model.global_known_tasks

        global_task = data_manager.n_task if self.async_task else local_task
        test_classes = np.arange(0, data_manager.get_cumulative_task_size(global_task))
        print("Global class:", test_classes.tolist())
        test_dataset = data_manager.get_dataset(
            test_classes, source="test", mode="test"
        )
        loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.global_net.eval()
        correct, total = 0, 0
        test_acc = 0.0
        targets_list, predicts_list = [], []
        prog_bar = tqdm(loader)
        for i, (_, inputs, targets) in enumerate(prog_bar):
            inputs = inputs.to(self.device)
            with torch.no_grad():
                model_outputs = self.global_net(inputs)
                outputs = (
                    model_outputs["logits"]
                    if isinstance(model_outputs, dict)
                    else model_outputs
                )
            class_mask = [
                col_index
                for col_index in range(outputs.shape[1])
                if col_index not in test_classes.tolist()
            ]
            outputs[:, class_mask] = float("-inf")
            predicts = torch.max(outputs, dim=1)[1]
            predicts = predicts.cpu()
            correct += (predicts == targets).sum()
            total += len(targets)
            targets_list.append(targets)
            predicts_list.append(predicts.cpu())

            test_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            info = "Test_accy {:.2f}".format(
                test_acc,
            )
            prog_bar.set_description(info)
        print(info)

        targets_list = torch.cat(targets_list, dim=0).numpy()
        predicts_list = torch.cat(predicts_list, dim=0).numpy()
        return test_acc, targets_list, predicts_list
