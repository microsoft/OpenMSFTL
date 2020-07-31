from typing import Dict, List
from ftl.agents import Client, Server
from .cifar import CIFAR10
from .mnist import MNIST
from .data_manager import DataManager


def process_data(data_config: Dict,
                 clients: List[Client],
                 server: Server) -> DataManager:
    data_set = data_config["data_set"]
    if data_set == 'cifar10':
        return CIFAR10(data_config=data_config,
                       clients=clients,
                       server=server).load_data()
    elif data_set == 'mnist':
        return MNIST(data_config=data_config,
                     clients=clients,
                     server=server).load_data()
    else:
        raise NotImplemented
