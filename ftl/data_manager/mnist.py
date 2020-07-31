from .data_manager import DataManager
from ftl.agents import Client, Server
from torchvision import datasets, transforms
from typing import Dict, List
import os

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class MNIST(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)
        self.mean = (0.1307,)
        self.std = (0.3081,)

    def load_data(self):
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=self.mean, std=self.std)])

        mnist_train = datasets.MNIST(root=root, download=self.download, train=True, transform=trans)
        mnist_test = datasets.MNIST(root=root, download=self.download, train=False, transform=trans)


