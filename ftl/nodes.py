import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader


class Client:
    def __init__(self, client_id, adv_noise=None):
        self.client_id = client_id
        self.local_train_data = None
        self.adversary_mode = adv_noise


class Server:
    def __init__(self):
        self.val_loader = None
        self.test_loader = None
