from ftl.agents import Client, Server
from ftl.training_utils import cycle
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from typing import Dict, List


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

    def _populate_data_partition_map(self):
        """ wrapper to Sampling data for client, server """
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

    def download_data(self) -> [datasets, datasets]:
        """ Downloads Data and Apply appropriate Transformations . returns train, test dataset """
        pass

    def distribute_data(self):
        _train_dataset, _test_dataset = self.download_data()
        self.server.test_loader = DataLoader(_test_dataset)

        # update data set stats
        total_train_samples = _train_dataset.data.shape[0]
        self.num_dev = int(self.dev_split * total_train_samples)
        self.num_train = total_train_samples - self.num_dev
        self.num_test = _test_dataset.data.shape[0]
        assert self.no_of_labels == len(_train_dataset.classes), 'Number of Labels of DataSet and Model Mismatch, ' \
                                                                 'fix passed arguments to change softmax dim'
        # partition data
        self._populate_data_partition_map()

        # populate server data loaders
        val_dataset = Subset(dataset=_train_dataset, indices=self.val_ix)
        self.server.val_loader = DataLoader(val_dataset.dataset, batch_size=self.batch_size)
        self.server.test_loader = DataLoader(_test_dataset, batch_size=self.batch_size)

        # populate client data loader
        for client in self.clients:
            local_dataset = Subset(dataset=_train_dataset, indices=self.data_distribution_map[client.client_id])
            client.local_train_data = DataLoader(local_dataset.dataset, shuffle=True, batch_size=self.batch_size)
            client.trainer.train_iter = iter(cycle(client.local_train_data))

