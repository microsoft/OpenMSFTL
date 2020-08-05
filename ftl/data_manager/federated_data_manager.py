from ftl.agents import Client, Server
from ftl.training_utils import cycle
from torchvision import datasets, transforms
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
        self.data_config = data_config
        # keep track of data distribution among clients
        self.clients = clients
        self.server = server

        # Data Set Properties to be populated / can be modified
        self.num_train = 0
        self.num_dev = 0
        self.num_test = 0
        self.val_ix = None
        self.data_distribution_map = {}
        self.download = self.data_config.get('download', True)

    @staticmethod
    def _compute_stats(train_dataset):
        try:
            mean = train_dataset.data.float().mean(axis=(0, 1, 2)) / 255
            std = train_dataset.data.float().std(axis=(0, 1, 2)) / 255
        except:
            mean = train_dataset.data.mean(axis=(0, 1, 2)) / 255
            std = train_dataset.data.std(axis=(0, 1, 2)) / 255
        return mean, std

    def _get_common_trans(self, _train_dataset):
        mean, std = self._compute_stats(_train_dataset)
        train_trans = transforms.Compose([transforms.RandomRotation(5),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomCrop(_train_dataset.data.shape[1], 4),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        return train_trans, test_trans

    def _populate_data_partition_map(self):
        """ wrapper to Sampling data for client, server """
        data_distribution_strategy = self.data_config.get("data_distribution_strategy", 'iid')
        if data_distribution_strategy == 'iid':
            self._iid_dist()
        else:
            raise NotImplemented

    def _iid_dist(self):
        """ Distribute the data iid into all the clients """
        all_indexes = np.arange(self.num_train + self.num_dev)

        # Let's assign points for Dev data
        if self.num_dev > 0:
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
        raise NotImplementedError("This method needs to be implemented")

    def distribute_data(self):
        """ Distributes Data among clients, Server accordingly. Makes ready to train-test """
        _train_dataset, _test_dataset = self.download_data()

        # update data set stats
        total_train_samples = _train_dataset.data.shape[0]
        self.num_dev = int(self.data_config.get('dev_split', 0.1) * total_train_samples)
        self.num_train = total_train_samples - self.num_dev
        self.num_test = _test_dataset.data.shape[0]

        assert self.data_config.get('num_labels') == len(_train_dataset.classes), \
            'Number of Labels of DataSet and Model output shape Mismatch, ' \
            'fix num_labels in client.config.data_config to change model output shape'

        if len(_train_dataset.data.shape) > 3:
            assert self.data_config.get('num_channels') == _train_dataset.data.shape[-1], \
                'Number of channels of DataSet and Model in channel shape Mismatch, ' \
                'fix num_channels in client.config.data_config to change model input shape'
        else:
            assert self.data_config.get('num_channels') == 1, \
                'Number of channels of DataSet and Model in channel shape Mismatch, ' \
                'fix num_channels in client.config.data_config to change model input shape'
        # partition data
        self._populate_data_partition_map()

        # populate server data loaders
        if self.val_ix:
            val_dataset = Subset(dataset=_train_dataset, indices=self.val_ix)
            self.server.val_loader = DataLoader(val_dataset.dataset,
                                                batch_size=self.data_config.get("infer_batch_size", 1),
                                                pin_memory=True,
                                                num_workers=1)

        self.server.test_loader = DataLoader(_test_dataset,
                                             batch_size=self.data_config.get("infer_batch_size", 1),
                                             pin_memory=True,
                                             num_workers=1)

        # populate client data loader
        for client in self.clients:
            local_dataset = Subset(dataset=_train_dataset,
                                   indices=self.data_distribution_map[client.client_id])
            client.local_train_data = DataLoader(local_dataset.dataset,
                                                 shuffle=True,
                                                 batch_size=client.client_opt_config.get("train_batch_size", 256),
                                                 pin_memory=True,
                                                 num_workers=2)
            client.trainer.train_iter = iter(cycle(client.local_train_data))
