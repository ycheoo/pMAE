import copy

import numpy as np
import torch
from Datasets.data_manager import DataManager
from scipy.spatial.distance import cdist
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.conf import get_device
from utils.toolkit import tensor2numpy

EPSILON = 1e-8


class BaseLearner(object):
    def __init__(self, client_id, args):
        self.client_id = client_id
        self.cur_task = -1
        self.cur_task_global = -1
        self.pre_task_global = -1
        self.n_known_class = 0
        self.n_total_class = 0
        self.network = None
        self.old_network = None
        self.topk = 5

        self.model = args.model

        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.tuned_epoch = args.tuned_epoch
        # self.sample_ratio = args.sample_ratio

        self.device = get_device(device_id=args.device_id)
        self.args = args

        self.train_loader = None

        self.batch_size = args.batch_size
        self.init_lr = args.init_lr
        self.weight_decay = (
            args.weight_decay if args.weight_decay is not None else 0.0005
        )
        self.min_lr = args.min_lr if args.min_lr is not None else 1e-8

        self.known_tasks = []
        self.known_classes = []
        self.known_available_classes = []

        self.cur_classes = []
        self.cur_available_classes = []

        self.new_task = False

        self.rehearsal = getattr(args, "rehearsal", False)
        self.memory_size_client = getattr(args, "memory_size_client", None)
        self.data_memory, self.targets_memory = np.array([]), np.array([])

        self.nme_classifier = getattr(args, "nme_classifier", False)

        self.except_part = args.freeze if hasattr(self.args, "freeze") else []

    @property
    def feature_dim(self):
        return self.network.feature_dim

    @property
    def total_classes(self):
        return self.known_classes + self.cur_classes

    @property
    def total_available_classes(self):
        return self.known_available_classes + self.cur_available_classes

    @property
    def curdata_size(self):
        assert self.train_loader is not None, "No Data."
        return len(self.train_loader.dataset)

    @property
    def samples_per_class(self):
        assert len(self.total_available_classes) != 0, "Total classes is 0"
        return self.memory_size_client // len(self.total_available_classes)

    @property
    def exemplar_size(self):
        assert len(self.data_memory) == len(self.targets_memory), "Exemplar size error."
        return len(self.targets_memory)

    def get_memory(self):
        if len(self.data_memory) == 0:
            return None
        else:
            return (self.data_memory, self.targets_memory)

    def check_loader(self, task, data_manager: DataManager):
        task_global = data_manager.get_task_id(task)
        cur_classes = data_manager.task_classlist[task_global]
        train_dataset = data_manager.get_dataset(
            np.array(cur_classes),
            source="train",
            mode="train",
        )
        return len(train_dataset) != 0

    def update_state(self, task, data_manager: DataManager):
        if self.cur_task != task:
            if task != 0:
                self.pre_task_global = self.cur_task_global
                self.known_classes.extend(self.cur_classes)
                self.known_available_classes.extend(self.cur_available_classes)
                self.cur_available_classes = []

            self.new_task = True
            self.cur_task = task
            self.cur_task_global = data_manager.get_task_id(self.cur_task)
            self.cur_classes = data_manager.task_classlist[self.cur_task_global]
            self.cur_available_classes = data_manager.get_available_label_categories(
                self.cur_classes, source="train"
            )

    def after_task(self, data_manager=None, global_epoch=None):
        self.new_task = False
        if self.rehearsal and (
            self.nme_classifier
            or (
                ((global_epoch + 1) * data_manager.n_task)
                % self.args.communication_epoch
                == 0
            )
        ):
            self.build_rehearsal_memory(data_manager, self.samples_per_class)

    def before_task(self, task, data_manager: DataManager):
        self.update_state(task, data_manager)
        if self.new_task:
            if not self.rehearsal:
                train_dataset = data_manager.get_dataset(
                    np.array(self.cur_classes),
                    source="train",
                    mode="train",
                )
            else:
                train_dataset = data_manager.get_dataset(
                    np.array(self.cur_classes),
                    source="train",
                    mode="train",
                    appendent=self.get_memory(),
                )
            self.train_dataset = train_dataset
            self.data_manager = data_manager
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return self.train_loader

    def train(self):
        pass

    def extract_vectors(self, loader):
        self.network.eval()
        vectors_list, targets_list = [], []

        with torch.no_grad():
            for _, inputs, targets in loader:
                targets = targets.numpy()
                vectors = tensor2numpy(
                    self.network.extract_vector(inputs.to(self.device))
                )

                vectors_list.append(vectors)
                targets_list.append(targets)

        return np.concatenate(vectors_list), np.concatenate(targets_list)

    def build_rehearsal_memory(self, data_manager, per_class):
        print("Available classes on client:", self.total_available_classes)
        self.reduce_exemplar(data_manager, per_class)
        self.construct_exemplar(data_manager, per_class)

    def reduce_exemplar(self, data_manager, m):
        dummy_data, dummy_targets = copy.deepcopy(self.data_memory), copy.deepcopy(
            self.targets_memory
        )
        self.class_means = np.zeros((len(self.total_classes), self.feature_dim))
        self.data_memory, self.targets_memory = np.array([]), np.array([])

        if len(self.known_available_classes) == 0:
            return

        info = "No exemplars to be reduced"
        print("Reducing exemplars...(max {} per classes)".format(m))
        prog_bar = tqdm(self.known_available_classes)
        for _, class_idx in enumerate(prog_bar):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            if len(dd) == 0:
                continue
            self.data_memory = (
                np.concatenate((self.data_memory, dd))
                if len(self.data_memory) != 0
                else dd
            )
            self.targets_memory = (
                np.concatenate((self.targets_memory, dt))
                if len(self.targets_memory) != 0
                else dt
            )
            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            vectors, _ = self.extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self.class_means[class_idx, :] = mean

            info = "Class {}, Exemplar {}".format(class_idx, len(idx_dataset))
            prog_bar.set_description(info)

        print(info)

    def construct_exemplar(self, data_manager, per_class):
        info = "No exemplars to construct"
        print("Constructing exemplars...(max {} per classes)".format(per_class))

        prog_bar = tqdm(self.cur_available_classes)
        for _, class_idx in enumerate(prog_bar):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            info = "Class {}, Exemplar {}".format(class_idx, len(idx_dataset))
            prog_bar.set_description(info)

            if len(data) == 0:
                continue

            m = min(len(data), per_class)
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            vectors, _ = self.extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self.data_memory = (
                np.concatenate((self.data_memory, selected_exemplars))
                if len(self.data_memory) != 0
                else selected_exemplars
            )
            self.targets_memory = (
                np.concatenate((self.targets_memory, exemplar_targets))
                if len(self.targets_memory) != 0
                else exemplar_targets
            )
            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self.extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self.class_means[class_idx, :] = mean

        print(info)
