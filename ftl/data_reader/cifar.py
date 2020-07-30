from .data_manager import DataManager
from typing import Dict, List
from ftl.agents import Client, Server


class CIFAR10(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)
