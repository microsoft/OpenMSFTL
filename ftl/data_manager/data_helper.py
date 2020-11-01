# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict, List
from ftl.agents import Client, Server
from .vision_datasets import FedMNIST, FedCIFAR10, FedFashionMNIST
from .federated_data_manager import JsonlDataManager

def distribute_data(data_config: Dict,
                 clients: List[Client],
                 server: Server):
    """
    Distribute data to server and clients
    """
    data_set = data_config["client"]["data_set"]
    if data_set == 'cifar10':
        dm = FedCIFAR10(data_config=data_config["client"],
                        clients=clients,
                        server=server)
    elif data_set == 'mnist':
        dm = FedMNIST(data_config=data_config["client"],
                      clients=clients,
                      server=server)
    elif data_set == 'fashion_mnist':
        dm = FedFashionMNIST(data_config=data_config["client"],
                             clients=clients,
                             server=server)
    elif data_set == 'sent140':
        # data_config["server"]["test"]["test_jsonl"] = "/blob/sdrg/user/didimit/Data/Projects/Leaf/data/sent140/data/test/all_data_0_0_keep_5_test_9.json"
        # data_config["server"]["val"]["val_jsonl"] = "/blob/sdrg/user/didimit/Data/Projects/Leaf/data/sent140/data/val/all_data_0_0_keep_5_val_9.json"
        # data_config["client"]["train"]["list_of_train_jsonls"] = "/blob/sdrg/user/didimit/Data/Projects/Leaf/data/sent140/data/train/all_data_0_0_keep_5_train_9.json"
        # data_config["client"]["train"]["vocab_dict"] = "/blob/sdrg/user/didimit/Data/Projects/Leaf/models/sent140/embs.json"
        dm = JsonlDataManager(data_config=data_config,
                              clients=clients,
                              server=server,
                              vec_size=300)
    else:
        raise NotImplemented

    dm.distribute_data()
    return dm.server, dm.clients
