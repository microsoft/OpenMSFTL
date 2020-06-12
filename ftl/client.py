import numpy as np
from typing import List


class Client:
    def __init__(self, client_id, adv_noise=None):
        self.client_id = client_id
        self.adversary_mode = adv_noise
        self.num_data_samples = 0


def distribute_data(clients: List[Client], num_samples: int):
    num_clients = len(clients)
    data_partition_ix = []
    num_samples_per_machine = num_samples // num_clients
    all_indexes = np.arange(num_samples)

    for machine in range(0, num_clients - 1):
        data_partition_ix += [
            all_indexes[num_samples_per_machine * machine: num_samples_per_machine * (machine + 1)]]

    # put the rest in the last machine
    data_partition_ix += [all_indexes[num_samples_per_machine * (num_clients - 1):]]
    print("All but last machine has {} data points".format(num_samples_per_machine))
    print("length of last machine indices:", len(data_partition_ix[-1]))

    return data_partition_ix, num_samples_per_machine
