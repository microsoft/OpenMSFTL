import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader


class Client:
    def __init__(self, client_id, adv_noise=None):
        self.client_id = client_id
        self.local_data = None
        self.adversary_mode = adv_noise


def distribute_data(data_loader: DataLoader, clients: List[Client], num_batches: int):
    data_partition_ix = _get_batch_partition_indices(clients=clients, num_batches=num_batches)
    for client, indices in data_partition_ix.items():
        pass


def _get_batch_partition_indices(clients: List[Client], num_batches: int) -> Dict[Client, List]:
    num_clients = len(clients)
    data_partition_ix = {}
    num_samples_per_machine = num_batches // num_clients
    all_indexes = list(np.arange(num_batches))

    for ix, client in enumerate(clients):
        data_partition_ix[client] = \
            all_indexes[num_samples_per_machine * ix: num_samples_per_machine * (ix + 1)]

    # append the rest in the last client
    data_partition_ix[clients[-1]].append(all_indexes[num_samples_per_machine * (num_clients - 1):])

    return data_partition_ix
