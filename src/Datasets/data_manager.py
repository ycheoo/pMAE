import random
from itertools import accumulate

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def select(x, y, idx):
    indices = np.where(y == idx)[0]
    return x[indices], y[indices]


def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


class DataManager(object):
    def __init__(
        self,
        dm_id,
        idata,
        init_class,
        inc_class,
        shuffle=False,
        is_global=False,
        async_task=False,
        class_order=None,
    ):
        self.setup_data(idata, shuffle, is_global, class_order)

        assert init_class <= self.get_total_classnum, "No enough classes."
        self.inc_classlist = [init_class]
        while sum(self.inc_classlist) + inc_class < self.get_total_classnum:
            self.inc_classlist.append(inc_class)
        offset_class = self.get_total_classnum - sum(self.inc_classlist)
        if offset_class > 0:
            self.inc_classlist.append(offset_class)

        self.task_order = list(range(self.n_task))
        self.task_classlist = [
            [
                class_id
                for class_id in range(
                    sum(self.inc_classlist[:i]), sum(self.inc_classlist[: i + 1])
                )
            ]
            for i in range(len(self.inc_classlist))
        ]
        if is_global:
            print(self.task_classlist)
        if async_task:
            random.shuffle(self.task_order)
            print(f"Client {dm_id} task order: {self.task_order}")

        # if not is_global:
        #     random.seed(seed + dm_id)
        #     sample_ratio = 0.6
        #     task_classlist = [
        #         random.sample(
        #             self.class_list[
        #                 sum(self.inc_classlist[:i]) : sum(self.inc_classlist[: i + 1])
        #             ],
        #             int(self.inc_classlist[i] * sample_ratio),
        #         )
        #         for i in range(len(self.inc_classlist))
        #     ]
        #     self.inc_classlist = [
        #         int(n_class * sample_ratio) for n_class in self.inc_classlist
        #     ]

        #     if async_task:
        #         random.shuffle(task_classlist)

        #     self.class_list = [
        #         class_item for classlist in task_classlist for class_item in classlist
        #     ]

        #     print("Final task order:", task_classlist)

    @property
    def n_task(self):
        return len(self.inc_classlist)

    @property
    def get_total_classnum(self):
        return len(self.class_order)

    def get_target_taskid(self, target):
        for task_id in range(self.n_task):
            target -= self.get_task_size(task_id)
            if target < 0:
                return task_id

    def get_task_id(self, task):
        return self.task_order[task]

    def get_task_size(self, task):
        return self.inc_classlist[task]

    def get_cumulative_task_size(self, task):
        return sum(self.inc_classlist[: task + 1])

    def setup_data(self, idata, shuffle, is_global, class_order):
        self.train_data, self.train_targets = idata.train_data, idata.train_targets
        self.test_data, self.test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        self.train_transform = idata.train_transform
        self.test_transform = idata.test_transform
        self.common_transform = idata.common_transform

        order = list(np.unique(self.train_targets))
        if shuffle and is_global:
            order = np.random.permutation(order).tolist()
            print("Class order shuffled", order)
        self.class_order = order if is_global else class_order
        # Map indices
        self.train_targets = map_new_class_index(self.train_targets, self.class_order)
        self.test_targets = map_new_class_index(self.test_targets, self.class_order)

    def get_available_label_categories(self, indices_class, source):
        x, y = self.train_data, self.train_targets
        targets = []
        for idx in range(len(indices_class)):
            class_data, class_targets = select(x, y, indices_class[idx])
            targets.append(class_targets)
        targets = np.concatenate(targets)
        return np.unique(targets).tolist()

    def get_dataset(self, indices_class, source, mode, appendent=None, ret_data=False):
        if source == "train":
            x, y = self.train_data, self.train_targets
        elif source == "test":
            x, y = self.test_data, self.test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            transform = transforms.Compose(
                [*self.train_transform, *self.common_transform]
            )
        elif mode == "test":
            transform = transforms.Compose(
                [*self.test_transform, *self.common_transform]
            )
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in range(len(indices_class)):
            class_data, class_targets = select(x, y, indices_class[idx])
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, transform, self.use_path)
        else:
            return DummyDataset(data, targets, transform, self.use_path)

    def get_dataset_with_split(
        self, indices_class, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self.train_data, self.train_targets
        elif source == "test":
            x, y = self.test_data, self.test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            transform = transforms.Compose(
                [*self.train_transform, *self.common_transform]
            )
        elif mode == "test":
            transform = transforms.Compose(
                [*self.test_transform, *self.common_transform]
            )
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in range(len(indices_class)):
            class_data, class_targets = select(x, y, indices_class[idx])
            if len(class_data) == 0:
                continue
            val_idx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_idx = list(set(np.arange(len(class_data))) - set(val_idx))
            val_data.append(class_data[val_idx])
            val_targets.append(class_targets[val_idx])
            train_data.append(class_data[train_idx])
            train_targets.append(class_targets[train_idx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            appendent_indices_class = list(set(append_targets))
            for idx in range(len(appendent_indices_class)):
                append_data, append_targets = select(
                    appendent_data, appendent_targets, appendent_indices_class[idx]
                )
                if len(append_data) == 0:
                    continue
                val_idx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_idx = list(set(np.arange(len(append_data))) - set(val_idx))
                val_data.append(append_data[val_idx])
                val_targets.append(append_targets[val_idx])
                train_data.append(append_data[train_idx])
                train_targets.append(append_targets[train_idx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, transform, self.use_path
        ), DummyDataset(val_data, val_targets, transform, self.use_path)


class DummyDataset(Dataset):
    def __init__(self, images, labels, transform, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.transform = transform
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.transform(pil_loader(self.images[idx]))
        else:
            image = self.transform(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label

    @property
    def label_categories(self):
        return np.unique(self.labels).tolist()


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
