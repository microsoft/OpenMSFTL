import numpy as np
from typing import List, Dict


class Client:
    def __init__(self, client_id, adv_noise=None):
        self.client_id = client_id
        self.local_data = None
        self.adversary_mode = adv_noise
        self.num_data_samples = 0


def distribute_data(clients: List[Client], num_samples: int) -> Dict[int, List]:
    num_clients = len(clients)
    data_partition_ix = {}
    num_samples_per_machine = num_samples // num_clients
    all_indexes = list(np.arange(num_samples))

    for ix, client in enumerate(clients):
        data_partition_ix[client.client_id] = \
            all_indexes[num_samples_per_machine * ix: num_samples_per_machine * (ix + 1)]

    # append the rest in the last client
    data_partition_ix[clients[-1].client_id].append(all_indexes[num_samples_per_machine * (num_clients - 1):])

    return data_partition_ix
