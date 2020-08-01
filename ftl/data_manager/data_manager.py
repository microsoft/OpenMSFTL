from typing import Dict, List
from ftl.agents import Client, Server
from torchvision import datasets
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
        self.val_ix = None
        self.data_distribution_map = {}
        self.no_of_labels = data_config["num_labels"]

    def fetch_data(self) -> [datasets, datasets]:
        """ Downloads Data and Apply appropriate Transformations . returns train, test dataset """
        pass

    def distribute_data(self):
        """ process train, test dataset and distribute among clients"""
        pass

    def _populate_data_partition_map(self):
        if self.data_distribution_strategy == 'iid':
            self._iid_dist()
        else:
            raise NotImplemented

    def _iid_dist(self):
        """ Distribute the data iid into all the clients """
        all_indexes = np.arange(self.num_train + self.num_dev)
        # Let's assign points for Dev data
        self.val_ix = set(np.random.choice(a=all_indexes, size=self.num_dev, replace=False))
        all_indexes = list(set(all_indexes) - self.val_ix)

        # split rest to clients for train
        num_clients = len(self.clients)
        num_samples_per_machine = self.num_train // num_clients

        for machine_ix in range(0, num_clients - 1):
            self.data_distribution_map[self.clients[machine_ix].client_id] = \
                set(np.random.choice(a=all_indexes, size=num_samples_per_machine, replace=False))
            all_indexes = list(set(all_indexes) - self.data_distribution_map[self.clients[machine_ix].client_id])
        # put the rest in the last machine
        self.data_distribution_map[self.clients[-1].client_id] = all_indexes

