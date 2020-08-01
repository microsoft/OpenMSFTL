from .data_manager import DataManager
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
        """ Downloads Data and Apply appropriate Transformations """
        mean = (0.1307,)
        std = (0.3081,)
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.MNIST(root=root, download=self.download, train=True, transform=trans)
        _test_dataset = datasets.MNIST(root=root, download=self.download, train=False, transform=trans)

        return _train_dataset, _test_dataset










