from .data_manager import DataManager
from typing import Dict, List
from ftl.agents import Client, Server
from torchvision import datasets, transforms
import os

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class CIFAR10(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_data(self):
        train_trans = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=self.mean, std=self.std),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomCrop(32, 4)])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=self.mean, std=self.std)])

        cifar_train = datasets.CIFAR10(root=root, download=self.download, train=True, transform=train_trans)
        cifar_test = datasets.CIFAR10(root=root, download=self.download, train=False, transform=test_transform)
