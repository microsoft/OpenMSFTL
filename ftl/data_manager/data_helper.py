# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict, List
from ftl.agents import Client, Server
from .vision_datasets import FedMNIST, FedCIFAR10, FedFashionMNIST
from .federated_data_manager import DataManager


def process_data(data_config: Dict,
                 clients: List[Client],
                 server: Server) -> DataManager:
    data_set = data_config["data_set"]
    if data_set == 'cifar10':
        return FedCIFAR10(data_config=data_config,
                          clients=clients,
                          server=server)
    elif data_set == 'mnist':
        return FedMNIST(data_config=data_config,
                        clients=clients,
                        server=server)
    elif data_set == 'fashion_mnist':
        return FedFashionMNIST(data_config=data_config,
                               clients=clients,
                               server=server)
    else:
        raise NotImplemented
