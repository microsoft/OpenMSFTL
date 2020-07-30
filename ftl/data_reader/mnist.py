from .data_manager import DataManager
from typing import Dict, List
from ftl.agents import Client, Server


class MNIST(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)
