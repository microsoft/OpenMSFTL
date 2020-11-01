# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from .federated_data_manager import TorchVisionDataManager
from ftl.agents import Client, Server
from torchvision import datasets, transforms
from torch.utils.data import Subset
from typing import Dict, List
import os
import numpy as np

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


def split_train_and_val_dataset(train_val_dataset, data_config):
    """
    Split the whole data into training and validation parts.
    """
    def _check_torchvision_dataset(labeled_dataset, data_config):
        """
        Check configuration of the dataset
        """
        assert data_config.get('num_labels') == len(labeled_dataset.classes), \
            'Number of Labels of DataSet and Model output shape Mismatch, ' \
            'fix num_labels in client.config.data_config to change model output shape'

        if len(labeled_dataset.data.shape) > 3:
            assert data_config.get('num_channels') == labeled_dataset.data.shape[-1], \
                'Number of channels of DataSet and Model in channel shape Mismatch, ' \
                'fix num_channels in client.config.data_config to change model input shape'
        else:
            assert data_config.get('num_channels') == 1, \
                'Number of channels of DataSet and Model in channel shape Mismatch, ' \
                'fix num_channels in client.config.data_config to change model input shape'

    _check_torchvision_dataset(train_val_dataset, data_config)
    total_train_samples = train_val_dataset.data.shape[0]
    all_indexes = np.arange(total_train_samples)
    # Create a validation dataset
    num_dev = int(data_config.get('dev_split', 0.1) * total_train_samples)
    val_indexes = set(np.random.choice(a=all_indexes, size=num_dev, replace=False))
    #val_dataset = Subset(dataset=train_val_dataset, indices=val_indexes)
    # Use the rest of data as a training dataset
    train_indexes = set(all_indexes) - val_indexes
    #train_dataset = Subset(dataset=train_val_dataset, indices=train_indexes)

    return list(train_indexes), list(val_indexes)


class FedMNIST(TorchVisionDataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        TorchVisionDataManager.__init__(self, data_config=data_config, clients=clients, server=server)

    def download_data(self) -> [datasets, datasets]:
        _train_val_dataset = datasets.MNIST(root=root, download=self.download, train=True)
        train_indexes, val_indexes = split_train_and_val_dataset(_train_val_dataset, self.data_config)
        # Compute the mean and std from raw training data
        mean, std = self._get_common_data_trans(_train_val_dataset) #Subset(dataset=_train_val_dataset, indices=train_indexes))
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        # Create transformed datasets
        _train_val_dataset = datasets.MNIST(root=root, download=self.download, train=True, transform=train_trans)
        _train_dataset = Subset(dataset=_train_val_dataset, indices=train_indexes)
        _val_dataset = Subset(dataset=_train_val_dataset, indices=val_indexes)
        _test_dataset = datasets.MNIST(root=root, download=self.download, train=False, transform=test_trans)

        return _train_dataset, _val_dataset, _test_dataset


class FedFashionMNIST(TorchVisionDataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        TorchVisionDataManager.__init__(self, data_config=data_config, clients=clients, server=server)

    def download_data(self) -> [datasets, datasets]:
        _train_val_dataset = datasets.FashionMNIST(root=root, download=self.download, train=True)
        train_indexes, val_indexes = split_train_and_val_dataset(_train_val_dataset, self.data_config)
        # Compute the mean and std from raw training data
        mean, std = self._get_common_data_trans(_train_val_dataset) #Subset(dataset=_train_val_dataset, indices=train_indexes))
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        # Create transformed datasets
        _train_val_dataset = datasets.FashionMNIST(root=root, download=self.download, train=True, transform=train_trans)
        _train_dataset = Subset(dataset=_train_val_dataset, indices=train_indexes)
        _val_dataset = Subset(dataset=_train_val_dataset, indices=val_indexes)
        _test_dataset = datasets.FashionMNIST(root=root, download=self.download, train=False, transform=test_trans)

        return _train_dataset, _val_dataset, _test_dataset


class FedCIFAR10(TorchVisionDataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        TorchVisionDataManager.__init__(self, data_config=data_config, clients=clients, server=server)

    def download_data(self) -> [datasets, datasets]:
        _train_val_dataset = datasets.CIFAR10(root=root, download=self.download, train=True)
        train_indexes, val_indexes = split_train_and_val_dataset(_train_val_dataset, self.data_config)
        # Compute the mean and std from raw training data
        mean, std = self._get_common_data_trans(_train_val_dataset) # Subset(dataset=_train_val_dataset, indices=train_indexes))
        train_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        # Create transformed datasets
        _train_val_dataset = datasets.CIFAR10(root=root, download=self.download, train=True, transform=train_trans)
        _train_dataset = Subset(dataset=_train_val_dataset, indices=train_indexes)
        _val_dataset = Subset(dataset=_train_val_dataset, indices=val_indexes)
        _test_dataset = datasets.CIFAR10(root=root, download=self.download, train=False, transform=test_trans)

        return _train_dataset, _test_dataset
