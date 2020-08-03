from typing import Dict, List
import numpy as np
from ftl.agents.client import Client
from .gar import FedAvg, SpectralFedAvg


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object and weights.
    """
    def __init__(self, aggregation_config: Dict):
        self.aggregation_config = aggregation_config
        self.gar = self.__get_gar()

    def __get_gar(self):
        if self.aggregation_config["aggregation_scheme"] == 'fed_avg':
            return FedAvg(aggregation_config=self.aggregation_config)
        elif self.aggregation_config["aggregation_scheme"] == 'fed_lr_avg':
            return SpectralFedAvg(aggregation_config=self.aggregation_config)
        else:
            raise NotImplementedError

    def get_aggregate(self, clients: List[Client], alphas: np.ndarray = None) -> np.array:
        if len(clients) == 0:
            raise Exception('Client List is Empty')
        # create a stacked Gradient Matrix G = [G0 | G1 | .... | Gn]'
        # each row corresponds to gradient (compressed) vector for each client
        G = np.zeros((len(clients), len(clients[0].grad)), dtype=clients[0].grad.dtype)
        for ix, client in enumerate(clients):
            G[ix, :] = client.C.compress(client.grad)
        agg_grad = self.gar.aggregate(G=G, alphas=alphas)
        return agg_grad

