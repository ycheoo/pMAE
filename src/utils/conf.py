import torch


def get_device(device_id) -> torch.device:
    return torch.device(
        "cuda:" + str(device_id) if torch.cuda.is_available() else "cpu"
    )


def data_path() -> str:
    return "~/data/"


def log_path() -> str:
    return "./logs/"


def result_path() -> str:
    return "./results/"


def config_path() -> str:
    return "./configs/"


def checkpoint_path() -> str:
    return "./checkpoints/"
