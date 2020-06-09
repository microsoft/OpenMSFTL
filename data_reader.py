import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
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
        :param data_set:   pass dataset name. Currently Supported:
                           mnist  :  http://yann.lecun.com/exdb/mnist/
                           cifar10:  https://www.cs.toronto.edu/~kriz/cifar.html
        :param batch_size: Choose Batch Size , default is 32
        :param download:   Whether to download the data. Default True
        :param dev_set:    Many datasets don't have dev set. If this flag is True,
                           return part of train set as dev
        :param split:      Choose what fraction of Train should be treated as dev, Only if dev_set flag is True
                           Default is 0.1
        """
        self.data_set = data_set
        self.download = download
        self.shape_of_feat = None
        self.no_of_labels = None
        self.batch_size = batch_size
        self.dev_flag = dev_set
        self.split = split

        if data_set == 'mnist':
            self.train_loader, self.test_loader = self._get_mnist()
        elif data_set == 'cifar10':
            self.train_loader, self.test_loader = self._get_cifar10()
        else:
            raise NotImplementedError

    def _get_mnist(self) -> [DataLoader, DataLoader]:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = datasets.MNIST(root=root, download=self.download, train=True, transform=trans)
        mnist_test = datasets.MNIST(root=root, download=self.download, train=False, transform=trans)

        train_loader = DataLoader(mnist_train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(mnist_test)
        self.no_of_labels = 10
        return train_loader, test_loader

    def _get_cifar10(self) -> [DataLoader, DataLoader]:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                (0.2023, 0.1994, 0.2010))])
        cifar_train = datasets.CIFAR10(root=root, download=self.download, train=True, transform=trans)
        cifar_test = datasets.CIFAR10(root=root, download=self.download, train=False, transform=trans)

        train_loader = DataLoader(cifar_train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(cifar_test)

        self.no_of_labels = 10
        return train_loader, test_loader
