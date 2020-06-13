import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader


class Client:
    def __init__(self, client_id, adv_noise=None):
        self.client_id = client_id
        self.local_data = None
        self.adversary_mode = adv_noise



