from typing import Dict, List
from ftl.agents import Client, Server
import torch
import numpy as np


class DataManager:
    """
    Base Class for all Data Readers
    """
    def __init__(self,
                 data_config: Dict,
                 clients: List[Client],
                 server: Server):

        torch.random.manual_seed(data_config["seed"])
        self.download = data_config["download"]
        self.batch_size = data_config["batch_size"]
        self.dev_split = data_config["dev_split"]
        self.data_distribution_strategy = data_config["data_dist_strategy"]

        # keep track of data distribution among clients
        self.clients = clients
        self.server = server

        # Data Set Properties to be populated / can be modified
        self.num_train = 0
        self.num_dev = 0
        self.num_test = 0
        self.no_of_labels = data_config["num_labels"]

    def load_data(self):
        pass

    def _populate_data_partition_map(self) -> Dict[int, List[int]]:
        data_distribution_map = {}
        if self.data_distribution_strategy == 'iid':
            self._iid_dist(data_distribution_map=data_distribution_map)
        else:
            raise NotImplemented
        return data_distribution_map

    def _iid_dist(self, data_distribution_map):
        """ Distribute the data iid into all the clients """
        num_clients = len(self.clients)
        num_samples_per_machine = self.num_train // num_clients
        all_indexes = np.arange(self.num_train)
        for machine_ix in range(0, num_clients - 1):
            data_distribution_map[self.clients[machine_ix].client_id] = \
                set(np.random.choice(a=all_indexes, size=num_samples_per_machine, replace=False))
            all_indexes = list(set(all_indexes) - data_distribution_map[self.clients[machine_ix].client_id])
        # put the rest in the last machine
        data_distribution_map[self.clients[-1].client_id] = all_indexes

