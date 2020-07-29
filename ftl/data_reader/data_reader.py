import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict
from ftl.agents.client import Client
import torch
import os
import numpy as np

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class DataReader:
    def __init__(self,
                 data_config: Dict,
                 clients: List[Client],
                 download: bool = True):
        """
        For More Info:
        This is the universal DataLoader Class. This can support all data
        fetching, appropriate pre-processing and mainly generates
        Train Validation and Test DataLoaders in appropriate
        PyTorch format ready to be used for Training ML models.
        Currently Supported Data sets:

        Computer Vision: mnist  :  http://yann.lecun.com/exdb/mnist/
                         cifar10:  https://www.cs.toronto.edu/~kriz/cifar.html

        """
        torch.random.manual_seed(data_config["seed"])
        self.data_set = data_config["data_set"]
        self.download = download
        self.batch_size = data_config["batch_size"]
        self.split = data_config["split"]
        self.do_sorting = data_config["do_sorting"]

        # keep track of data distribution among clients
        self.clients = clients
        self.data_distribution_map = {}

        # Data Set Properties to be populated
        self.num_train = 0
        self.num_dev = 0
        self.num_test = 0
        self.no_of_labels = 0

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        if self.data_set == 'mnist':
            self._get_mnist()
        elif self.data_set == 'cifar10':
            self._get_cifar10()
        else:
            raise NotImplementedError

    def _get_mnist(self):
        """
        Wrapper to Download (if flag = True) and pre-process MNIST data set
        :returns Train, Validation and Test DataLoaders
        """
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),
                                                         (0.3081,))])
        self.no_of_labels = 10

        # Handle Train and Dev
        # --------------------------------------
        mnist_train = datasets.MNIST(root=root, download=self.download, train=True,
                                     transform=trans)
        # compute number of data points
        self.num_dev = int(self.split * mnist_train.data.shape[0])
        self.num_train = mnist_train.data.shape[0] - self.num_dev
        self._distribute_data(data_set=mnist_train)

        # Handle Test
        # --------------------------------------
        mnist_test = datasets.MNIST(root=root, download=self.download, train=False, transform=trans)
        x_test = mnist_test.data
        y_test = mnist_test.targets
        self.test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=self.batch_size)
        self.num_test = mnist_test.data.shape[0]

    def _get_cifar10(self):
        """
        Wrapper to Download (if flag = True) and pre-process CIFAR10 data set
        :returns Train, Validation and Test DataLoaders
        """
        self.no_of_labels = 10
        # Handle Train and Dev
        # --------------------------------------
        train_trans = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2023, 0.1994, 0.2010))])
        # ,transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()])
        cifar_train = datasets.CIFAR10(root=root, download=self.download, train=True, transform=train_trans)
        self.num_dev = int(self.split * cifar_train.data.shape[0])
        self.num_train = cifar_train.data.shape[0] - self.num_dev
        self._distribute_data(data_set=cifar_train)

        # Handle Test
        # --------------------------------------
        test_trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010))])
        cifar_test = datasets.CIFAR10(root=root, download=self.download, train=False, transform=test_trans)
        self.num_test = cifar_test.data.shape[0]
        x_test = cifar_test.data
        y_test = cifar_test.targets
        # swap axis
        x_test = np.moveaxis(x_test, -1, 1)
        self.test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test),
                                                    torch.from_numpy(np.array(y_test))),
                                      batch_size=self.batch_size)  # We don't need to partition this

    def _distribute_data(self, data_set):
        """
        This is a util function to split a given torch data set into two
        based on the supplied split fraction. Useful for Train: Validation split

        :param data_set: provide the Torch data set you need to split
        :return: Returns two DataLoader object, Training, Validation
        """
        x = data_set.data
        y = data_set.targets

        if not isinstance(x, np.ndarray):
            x = x.numpy()
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        x_train = x[0: self.num_train, :, :]
        y_train = y[0:self.num_train]

        if self.do_sorting:
            y_sorted_ix = np.argsort(y_train)
            x_train = x_train[y_sorted_ix]
            y_train = y_train[y_sorted_ix]

        # Validation data goes into Aggregator only so need not distribute
        x_val = x[self.num_train:, :, :]
        y_val = y[self.num_train:]

        # if tensor of dim > 3 i.e. num channels > 1 handle channel
        if len(x_val.shape) > 3:
            print('Swapping axis, Torch convolutions expects channel first')
            x_val = np.moveaxis(x_val, -1, 1)

        self.val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val),
                                                   torch.from_numpy(y_val)),
                                     batch_size=self.batch_size,
                                     shuffle=True)

        # Now lets distribute the training data among clients
        self.data_distribution_map = self._get_data_partition_indices()  # populates the ix map

        for client in self.clients:
            local_indices = self.data_distribution_map[client.client_id]
            x_local = x_train[local_indices, :, :]
            y_local = y_train[local_indices]

            # if tensor of dim > 3 i.e. num channels > 1 handle channel
            if len(x_local.shape) > 3:
                x_local = np.moveaxis(x_local, -1, 1)

            client.local_train_data = DataLoader(TensorDataset(torch.from_numpy(x_local),
                                                               torch.from_numpy(y_local)),
                                                 batch_size=self.batch_size,
                                                 shuffle=True)

    def _get_data_partition_indices(self) -> Dict[Client, List[int]]:
        num_clients = len(self.clients)
        num_samples_per_machine = self.num_train // num_clients

        # Create a map of client -> ix of data assigned to that client
        client_data_map = {}
        all_indexes = np.arange(self.num_train)

        for machine_ix in range(0, num_clients - 1):
            client_data_map[self.clients[machine_ix].client_id] = \
                all_indexes[num_samples_per_machine * machine_ix: num_samples_per_machine * (machine_ix + 1)]

        # put the rest in the last machine
        client_data_map[self.clients[-1].client_id] = all_indexes[num_samples_per_machine * (num_clients - 1):]

        return client_data_map
