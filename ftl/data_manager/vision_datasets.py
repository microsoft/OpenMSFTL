# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from .federated_data_manager import DataManager
from ftl.agents import Client, Server
from torchvision import datasets, transforms
from typing import Dict, List
import os

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class FedMNIST(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)

    def download_data(self) -> [datasets, datasets]:
        _train_dataset = datasets.MNIST(root=root, download=self.download, train=True)
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.MNIST(root=root, download=self.download, train=True, transform=train_trans)
        _test_dataset = datasets.MNIST(root=root, download=self.download, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class FedFashionMNIST(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)

    def download_data(self) -> [datasets, datasets]:
        _train_dataset = datasets.FashionMNIST(root=root, download=self.download, train=True)
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.FashionMNIST(root=root, download=self.download, train=True, transform=train_trans)
        _test_dataset = datasets.FashionMNIST(root=root, download=self.download, train=False, transform=test_trans)

        return _train_dataset, _test_dataset


class FedCIFAR10(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)

    def download_data(self) -> [datasets, datasets]:
        _train_dataset = datasets.CIFAR10(root=root, download=self.download, train=True)
        mean, std = self._get_common_data_trans(_train_dataset)
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.CIFAR10(root=root, download=self.download, train=True, transform=train_trans)
        _test_dataset = datasets.CIFAR10(root=root, download=self.download, train=False, transform=test_trans)

        return _train_dataset, _test_dataset
