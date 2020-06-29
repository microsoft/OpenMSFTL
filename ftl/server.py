from ftl.client import Client
from ftl.optimization import Optimization, _get_lr
import copy
import torch
from typing import List
import numpy as np
import random


class Server:
    def __init__(self,
                 args,
                 model,
                 aggregation_scheme='fed_avg',
                 clients: List[Client] = None,
                 val_loader=None,
                 test_loader=None):

        self.args = args
        # Server has access to Test and Dev Data Sets to evaluate Training Process
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Server has a pointer to all clients
        self.clients = clients
        self.aggregation_scheme = aggregation_scheme

        # Server keeps track of model architecture and updated weights and grads at each round
        self.global_model = model
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.global_model.parameters()])
        self.client_grads = np.empty((len(self.clients), len(self.w_current)), dtype=self.w_current.dtype)

        # Containers to store metrics
        self.test_acc = []
        self.val_acc = []
        self.train_loss = []

    def init_client_models(self):
        # Loop over all clients and update the current params
        for client in self.clients:
            client.w_init = copy.deepcopy(self.w_current)

    def train_client_models(self, k, epoch):
        # Sample Clients to Train this round
        sampled_clients = random.sample(population=self.clients, k=k)

        # Now we will loop through these clients and do training steps
        for client in sampled_clients:
            client.train_step()



    def aggregate_client_updates(self, clients):
        """
        :param clients: Takes in a set of client compute nodes to aggregate
        :return: Updates the global model in the server with the aggregated parameters of the local models
        """
        if self.aggregation_scheme == 'fed_avg':
            agg_params = self._fed_average(clients=clients)
        else:
            raise NotImplementedError

        self.global_model.load_state_dict(agg_params)

    # ----------------------- #
    # Aggregation Strategies  #
    # ----------------------- #

    @staticmethod
    def _fed_average(clients: List[Client]):
        """
        Implements simple Fed Avg aggregation as introduced in:
        McMahan et.al. 'Communication-Efficient Learning of Deep Networks from Decentralized Data'
        http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
        """
        epoch_params = []
        for client in clients:
            epoch_params.append(copy.deepcopy(client.local_model.state_dict()))
        server_param = copy.deepcopy(epoch_params[0])
        for key in server_param.keys():
            for i in range(1, len(epoch_params)):
                server_param[key] += epoch_params[i][key]
            server_param[key] = torch.div(server_param[key], len(epoch_params))
        return server_param

    @staticmethod
    def _median(clients: List[Client]):
        raise NotImplementedError




