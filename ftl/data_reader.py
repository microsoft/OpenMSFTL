import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from typing import List
from ftl.client import Client
import torch
import os
import numpy as np

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class DataReader:
    def __init__(self,
                 data_set: str,
                 clients: List[Client],
                 batch_size: int = 32,
                 download: bool = True,
                 split: float = 0.1,
                 random_seed: int = 1):
        """
        For More Info:
        This is the universal DataLoader Class. This can support all data
        fetching, appropriate pre-processing and mainly generates
        Train Validation and Test DataLoaders in appropriate
        PyTorch format ready to be used for Training ML models.
        Currently Supported Data sets:

        Computer Vision: mnist  :  http://yann.lecun.com/exdb/mnist/
                         cifar10:  https://www.cs.toronto.edu/~kriz/cifar.html

        :param data_set:   pass data set name. One from the above list of supported Data sets.
        :param batch_size: Choose Batch Size , default is 32
        :param download:   Whether to download the data. Default True
        :param split:      Choose what fraction of Train should be treated as dev, Only if dev_set flag is True
                           Default is 0.1
        """
        torch.random.manual_seed(random_seed)
        self.data_set = data_set
        self.download = download
        self.batch_size = batch_size
        self.split = split

        self.data_partition_ix = {}  # keep track of data distribution among clients
        # Data Set Properties to be populated
        self.num_train = 0
        self.num_dev = 0
        self.num_test = 0
        self.no_of_labels = 0

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if data_set == 'mnist':
            self._get_mnist()
        elif data_set == 'cifar10':
            self._get_cifar10()
        else:
            raise NotImplementedError

    def _get_mnist(self):
        """
        Wrapper to Download (if flag = True) and pre-process MNIST data set
        :returns Train, Validation and Test DataLoaders
        """
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.no_of_labels = 10
        mnist_train = datasets.MNIST(root=root, download=self.download, train=True, transform=trans)
        mnist_test = datasets.MNIST(root=root, download=self.download, train=False, transform=trans)
        self.test_loader = DataLoader(mnist_test)  # We don't need to partition this

        # compute number of data points
        self.num_dev = int(self.split * mnist_train.data.shape[0])
        self.num_train = mnist_train.data.shape[0] - self.num_dev
        self.num_test = mnist_test.data.shape[0]

        self._split_torch_data(data_set=mnist_train, batch_size=self.batch_size)

    def _get_cifar10(self) -> [DataLoader, DataLoader, DataLoader]:
        """
        Wrapper to Download (if flag = True) and pre-process CIFAR10 data set
        :returns Train, Validation and Test DataLoaders
        """
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                (0.2023, 0.1994, 0.2010))])
        self.no_of_labels = 10
        cifar_train = datasets.CIFAR10(root=root, download=self.download, train=True, transform=trans)
        cifar_test = datasets.CIFAR10(root=root, download=self.download, train=False, transform=trans)
        test_loader = DataLoader(cifar_test)

        self.num_dev = int(self.split * cifar_train.data.shape[0])
        self.num_train = cifar_train.data.shape[0] - self.num_dev
        self.num_test = cifar_test.data.shape[0]

        # self._split_torch_data(data_set=cifar_train, batch_size=self.batch_size)


    def _split_torch_data(self, data_set, batch_size: int):
        """
        This is a util function to split a given torch data set into two
        based on the supplied split fraction. Useful for Train: Validation split

        :param data_set: provide the Torch data set you need to split
        :param batch_size: specify batch size for iterator
        :return: Returns two DataLoader object, Training, Validation
        """

        x = data_set.train_data.numpy()
        y = data_set.train_labels.numpy()

        x_train = x[0: self.num_train, :, :]
        y_train = y[0:self.num_train]

        # Validation data goes into Aggregator only so need not distribute
        x_val = torch.from_numpy(x[self.num_train:, :, :])
        y_val = torch.from_numpy(y[self.num_train:])
        self.val_loader = DataLoader(TensorDataset(x_val, y_val))

        # Now lets distribute the training data among clients

    def _get_data_partition_indices(self, clients: List[Client], num_batches: int):
        num_clients = len(clients)

        num_samples_per_machine = num_batches // num_clients
        all_indexes = list(np.arange(num_batches))

        for ix, client in enumerate(clients):
            self.data_partition_ix[client] = \
                all_indexes[num_samples_per_machine * ix: num_samples_per_machine * (ix + 1)]

        # append the rest in the last client
        self.data_partition_ix[clients[-1]].append(all_indexes[num_samples_per_machine * (num_clients - 1):])

        return self.data_partition_ix
