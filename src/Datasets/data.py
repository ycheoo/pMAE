import numpy as np
from torchvision import datasets, transforms
from utils.conf import data_path
from utils.toolkit import split_images_labels


class iData(object):
    def __init__(self) -> None:
        self.train_transform = []
        self.test_transform = []
        self.common_transform = []
        self.use_path = False
        self.train_data = None
        self.train_targets = None
        self.test_data = None
        self.test_targets = None

    def download_data(self):
        pass

    def load_data(self):
        pass

    def load_data(self, train_data, train_targets, test_data, test_targets):
        self.train_data, self.train_targets = train_data, train_targets
        self.test_data, self.test_targets = test_data, test_targets


def split_dataset(dataset):
    samples = []
    targets = []
    for item in dataset:
        samples.append(item[0])
        targets.append(item[1])

    return np.array(samples), np.array(targets)


def reduce_inst(seed, limit_inst, data, targets, class_order):
    np.random.seed(seed)
    reduced_data = []
    reduced_targets = np.repeat(class_order, repeats=limit_inst)
    for cls in class_order:
        indices = np.where(targets == cls)[0]
        subdata = data[indices]
        assert limit_inst <= len(subdata), "No enough instances."
        random_indices = np.random.choice(len(subdata), size=limit_inst, replace=False)
        reduced_data.append(subdata[random_indices])
    reduced_data = np.concatenate(reduced_data)
    return reduced_data, reduced_targets


def build_transform(is_train):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3.0 / 4.0, 4.0 / 3.0)

        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BICUBIC
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())

    return t


class iImageNetR(iData):
    def __init__(self):
        super().__init__()
        self.n_class = 200
        self.use_path = True

        self.train_transform = build_transform(True)
        self.test_transform = build_transform(False)
        self.common_transform = [
            # transforms.ToTensor(),
        ]

    def download_data(self):
        train_dir = f"{data_path()}/imagenet-r/train/"
        test_dir = f"{data_path()}/imagenet-r/test/"
        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iCUB(iData):
    def __init__(self):
        super().__init__()
        self.n_class = 200
        self.use_path = True

        self.train_transform = build_transform(True)
        self.test_transform = build_transform(False)
        self.common_transform = [
            # transforms.ToTensor(),
        ]

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = f"{data_path()}/cub/train/"
        test_dir = f"{data_path()}/cub/test/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
