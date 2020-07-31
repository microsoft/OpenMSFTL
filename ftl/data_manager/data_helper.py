from typing import Dict, List
from ftl.agents import Client, Server
from .cifar import CIFAR10
from .mnist import MNIST


def get_data_loader(data_config: Dict,
                    clients: List[Client],
                    server: Server):
    data_set = data_config["data_set"]
    if data_set == 'cifar10':
        return CIFAR10(data_config=data_config,
                       clients=clients,
                       server=server)
    elif data_set == 'mnist':
        return MNIST(data_config=data_config,
                     clients=clients,
                     server=server)
    else:
        raise NotImplemented

