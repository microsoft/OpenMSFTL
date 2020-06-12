import numpy as np
from typing import List


class Client:
    def __init__(self, client_id, adv_noise=None):
        self.client_id = client_id
        self.adversary_mode = adv_noise
        self.num_data_samples = 0


def distribute_data(clients: List[Client], num_samples: int):
    num_clients = len(clients)
    data_partition_ix = {}
    num_samples_per_machine = num_samples // num_clients
    all_indexes = np.arange(num_samples)

    for ix in range(0, num_clients - 1):
        data_partition_ix[clients[ix].client_id] = \
            [all_indexes[num_samples_per_machine * ix: num_samples_per_machine * (ix + 1)]]

    # put the rest in the last client
    data_partition_ix[clients[-1].client_id] = [all_indexes[num_samples_per_machine * (num_clients - 1):]]

    return data_partition_ix
