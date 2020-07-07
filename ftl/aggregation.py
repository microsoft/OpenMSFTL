from ftl.models import dist_weights_to_model, add_grads_to_model
from ftl.optimization import SchedulingOptimization
from ftl.client import Client
import numpy as np
import torch.nn as nn
from typing import Dict, List


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object and weights.
    """

    def __init__(self, agg_strategy: str,
                 model: nn.Module,
                 dual_opt_alg: str = None,
                 opt_group: Dict = None,
                 max_grad_norm: float = None):
        """
        :param agg_strategy: aggregation strategy, default to "fed_avg"
        :param model: class:`nn.Module`, the global model
        :param dual_opt_alg: type of (adaptive) Dual optimizer; see examples:  ftl/optimization.py
        :param opt_group: parameters for the optimizer; see details for ftl/optimization.py
        :param max_grad_norm: max norm of the gradients for gradient clipping, default to None
        """
        if opt_group is None:
            opt_group = {}
        self.agg_strategy = agg_strategy
        self.model = model

        # Instantiate the optimizer for an aggregator
        if dual_opt_alg is None or dual_opt_alg == "None":
            # simple model averaging
            self.optimizer = None
            self.lr_scheduler = None
        else:
            # dual optimization
            server_opt = SchedulingOptimization(model=model,
                                                opt_alg=dual_opt_alg,
                                                opt_group=opt_group)
            self.optimizer = server_opt.optimizer
            self.lr_scheduler = server_opt.lr_scheduler

        self.max_grad_norm = max_grad_norm
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

    def set_lr(self, current_lr):
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

    def update_model(self, clients: List[Client], current_lr: float = None) -> np.array:
        """
        Update a model with aggregated gradients
        :param current_lr:
        :param clients: A set of client compute nodes
        :return: The weights of the updated global model
        """

        if self.agg_strategy is 'fed_avg':
            self.__fed_avg(clients=clients, current_lr=current_lr)
            # update the model params with these weights
            dist_weights_to_model(weights=self.w_current, parameters=self.model.parameters())
        else:
            raise NotImplementedError
        return self.w_current

    def __fed_avg(self, clients, current_lr: float):
        """
        This implements two flavors the Federated Averaging GAR:
            a.  Simple FedAvg aggregation as introduced in:
                McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data",
                http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
            b.  FedAvg with dual optimization described in:
                D. Dimitriadis et al., "On a Federated Approach for Training Acoustic Models ", Interspeech 2021,
                S. J. Reddi et al., "Adaptive Federated Optimization", arXiv:2003.00295

        :param clients:
        :param current_lr:
        :return:
        """
        if self.optimizer is None:
            """ perform the Vanilla FedAvg """
            # client_grads = np.empty((len(clients), len(self.w_current)), dtype=self.w_current.dtype)
            # for ix, client in enumerate(clients):
            #     client_grads[ix, :] = client.grad
            lr = current_lr if current_lr is not None else 1.0
            # self.w_current -= lr * np.mean(client_grads, axis=0)
            agg_grad = self.__weighted_average(clients=clients)
            self.w_current -= lr * agg_grad
        else:
            """ Perform Dual Optimization """
            # change the learning rate of the optimizer
            if current_lr is not None:
                self.set_lr(current_lr)
            # set gradients to the model instance
            for ix, client in enumerate(clients):
                add_grads_to_model(grads=client.grad,
                                   parameters=self.model.parameters(),
                                   init=True if ix == 0 else False)
            # apply gradient clipping
            if self.max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                     self.max_grad_norm)

            # do optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # get the model weights
            self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

    @staticmethod
    def __weighted_average(clients, alphas=None):
        """ Implements weighted average of vectors if no weights are supplied
        then its equivalent to simple average / Fed Avg."""
        if alphas is None:
            alphas = [1] * len(clients)

        agg_grad = np.zeros_like(clients[0].grad)
        tot = np.sum(alphas)

        for alpha, client in zip(alphas, clients):
            agg_grad += (alpha / tot) * client

        return agg_grad

    @staticmethod
    def __geo_median(points):
        """ Computes Geometric median of given points (vectors) using Weiszfeld's Algorithm """
        pass


