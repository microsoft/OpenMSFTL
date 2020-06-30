import numpy as np


class Aggregator:
    def __init__(self, agg_strategy):
        self.agg_strategy = agg_strategy

    def compute_grad(self, clients, client_grads):
        # collect gradients from all clients
        for ix, client in enumerate(clients):
            client_grads[ix, :] = client.grad

        if self.agg_strategy is 'fed_avg':
            return self.fed_avg(client_grads=client_grads)
        else:
            raise NotImplementedError

    @staticmethod
    def fed_avg(client_grads):
        """
        Implements simple Fed Avg aggregation as introduced in:
        McMahan et.al. 'Communication-Efficient Learning of Deep Networks from Decentralized Data'
        http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
        """
        return np.mean(client_grads, axis=0)
