import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Optional
import torch
import os
curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class DataReader:
    def __init__(self,
                 data_set: str,
                 batch_size: int = 32,
                 download: bool = True,
                 dev_set: bool = True,
                 split: float = 0.1):
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
        :param dev_set:    Many data sets don't have dev set. If this flag is True,
                           return part of train set as dev
        :param split:      Choose what fraction of Train should be treated as dev, Only if dev_set flag is True
                           Default is 0.1
        """
        self.data_set = data_set
        self.download = download
        self.no_of_labels = None
        self.batch_size = batch_size
        self.dev_flag = dev_set
        self.split = split

        if data_set == 'mnist':
            self.train_loader, self.val_loader, self.test_loader = self._get_mnist()
        elif data_set == 'cifar10':
            self.train_loader, self.val_loader, self.test_loader = self._get_cifar10()
        else:
            raise NotImplementedError

    def _get_mnist(self) -> [DataLoader, DataLoader, DataLoader]:
        """
        Wrapper to Download (if flag = True) and pre-process MNIST data set
        :returns Train, Validation and Test DataLoaders
        """
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.no_of_labels = 10
        mnist_train = datasets.MNIST(root=root, download=self.download, train=True, transform=trans)
        mnist_test = datasets.MNIST(root=root, download=self.download, train=False, transform=trans)
        test_loader = DataLoader(mnist_test)

        if self.dev_flag:
            train_loader, dev_loader = self.split_torch_data(data_set=mnist_train, split=self.split,
                                                             batch_size=self.batch_size)
        else:
            train_loader, dev_loader = self.split_torch_data(data_set=mnist_train, split=0,
                                                             batch_size=self.batch_size)

        return train_loader, dev_loader, test_loader

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

        if self.dev_flag:
            train_loader, dev_loader = self.split_torch_data(data_set=cifar_train, split=self.split,
                                                             batch_size=self.batch_size)
        else:
            train_loader, dev_loader = self.split_torch_data(data_set=cifar_train, split=0,
                                                             batch_size=self.batch_size)
        return train_loader, dev_loader, test_loader

    @staticmethod
    def split_torch_data(data_set, split: float, batch_size: int) -> [DataLoader, DataLoader]:
        """
        This is a util function to split a given torch data set into two based on the supplied
        split fraction. Useful for Train: Validation split

        :param data_set: provide the Torch data set you need to split
        :param split: specify what fraction of data should be validation
        :param batch_size: specify batch size for iterator
        :return: Returns two DataLoader object, Training, Validation
        """
        # Get Number of samples in train and validation bucket based on split
        no_of_dev = int(split * data_set.data.shape[0])
        no_of_train = data_set.data.shape[0] - no_of_dev

        # split the data set randomly into train and dev buckets and get indices
        train_set, val_set = torch.utils.data.random_split(data_set, [no_of_train, no_of_dev])
        train_ix = train_set.indices
        val_ix = val_set.indices

        train_sampler = SubsetRandomSampler(train_ix)
        val_sampler = SubsetRandomSampler(val_ix)

        train_loader = DataLoader(dataset=data_set, sampler=train_sampler, batch_size=batch_size)
        val_loader = DataLoader(dataset=data_set, sampler=val_sampler, batch_size=batch_size)

        return train_loader, val_loader
