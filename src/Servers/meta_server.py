import numpy as np
import torch.nn as nn
from utils.conf import get_device


class SeverMethod(nn.Module):
    """
    Federated learning Methods.
    """

    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        self.device = get_device(device_id=self.args.device_id)
        self.global_known_tasks = []
        self.global_known_classes = []

    def sever_update(self):
        pass
