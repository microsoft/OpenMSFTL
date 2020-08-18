# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict, List
import numpy as np
from ftl.agents.client import Client
from .gar import FedAvg, SpectralFedAvg, MinLoss
from sklearn.utils.extmath import randomized_svd


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object and weights.
    """

    def __init__(self, aggregation_config: Dict):
        self.aggregation_config = aggregation_config
        self.gar = self.__get_gar()
        self.curr_G = None

    def __get_gar(self):
        if self.aggregation_config["aggregation_scheme"] == 'fed_avg':
            return FedAvg(aggregation_config=self.aggregation_config)
        elif self.aggregation_config["aggregation_scheme"] == 'fed_spectral_avg':
            return SpectralFedAvg(aggregation_config=self.aggregation_config)
        elif self.aggregation_config["aggregation_scheme"] == 'min_loss':
            return MinLoss(aggregation_config=self.aggregation_config)
        else:
            raise NotImplementedError

    def aggregate_grads(self, clients: List[Client], client_losses: List[float], analyze_pc: bool = True) -> np.array:
        if len(clients) == 0:
            raise Exception('Client List is Empty')
        G = np.zeros((len(clients), len(clients[0].grad)), dtype=clients[0].grad.dtype)
        for ix, client in enumerate(clients):
            G[ix, :] = client.C.compress(client.grad)
        if analyze_pc:
            print("Randomized SVD of G for Spectral Analysis")
            U, S, V = randomized_svd(G, n_components=min(G.shape[0], G.shape[1]), flip_sign=True)
            self.gar.Sigma_tracked.append(S)
        self.curr_G = G
        client_ids = np.array([c.client_id for c in clients])
        agg_grad = self.gar.aggregate(G=G, client_ids=client_ids, losses=client_losses)

        return agg_grad

