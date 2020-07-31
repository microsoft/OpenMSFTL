from typing import Dict, List
from ftl.agents import Client, Server
from .cifar import CIFAR10
from .mnist import MNIST


def process_data(data_config: Dict,
                 clients: List[Client],
                 server: Server):
    data_set = data_config["data_set"]
    if data_set == 'cifar10':
        CIFAR10(data_config=data_config,
                clients=clients,
                server=server).load_data()
    elif data_set == 'mnist':
        MNIST(data_config=data_config,
              clients=clients,
              server=server).load_data()
    else:
        raise NotImplemented
