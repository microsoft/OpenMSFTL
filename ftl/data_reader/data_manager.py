from typing import Dict, List
from ftl.agents import Client, Server
import torch


class DataManager:
    """
    Base Class for all Data Readers
    """
    def __init__(self,
                 data_config: Dict,
                 clients: List[Client],
                 server: Server):

        torch.random.manual_seed(data_config["seed"])
        self.download = data_config["download"]
        self.batch_size = data_config["batch_size"]
        self.dev_split = data_config["dev_split"]

        # keep track of data distribution among clients
        self.clients = clients
        self.server = server
        self.data_distribution_map = {}

        # Data Set Properties to be populated / can be modified
        self.num_train = 0
        self.num_dev = 0
        self.num_test = 0
        self.no_of_labels = data_config["num_labels"]

    def load_data(self):
        pass
