from ftl.models import dist_weights_to_model, dist_grads_to_model
from ftl.optimization import SchedulingOptimization
from ftl.client import Client
import torch.nn as nn
from typing import Dict, List
from collections import defaultdict
import numpy as np


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object and weights.
    """
    def __init__(self, agg_strategy: str,
                 model: nn.Module,
                 dual_opt_alg: str = "Adam",
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
        server_opt = SchedulingOptimization(model=model, opt_alg=dual_opt_alg, opt_group=opt_group)
        self.optimizer = server_opt.optimizer
        self.lr_scheduler = server_opt.lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

    def set_lr(self, current_lr):
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

    def update_model(self, clients: List[Client], current_lr: float = 0.01) -> np.array:
        """
        Update server model with aggregated gradients
        :param current_lr:
        :param clients: A set of client compute nodes
        :return: The weights of the updated global model
        """
        if self.agg_strategy is 'fed_avg':
            agg_grad = self.__fed_avg(clients=clients)
        else:
            raise NotImplementedError

        # Now Run a step of Dual Optimizer (Server Step)
        self.__server_step(agg_grad=agg_grad, current_lr=current_lr)
        dist_weights_to_model(weights=self.w_current, parameters=self.model.parameters())
        return self.w_current

    def __server_step(self, agg_grad, current_lr: float):
        """
        Implements Server (Dual) Optimization with the option to use Adaptive Optimization
        D. Dimitriadis et al., "On a Federated Approach for Training Acoustic Models ", Interspeech 2021,
        S. J. Reddi et al., "Adaptive Federated Optimization", arXiv:2003.00295
        """
        self.set_lr(current_lr)
        # Update the model with aggregated gradient
        dist_grads_to_model(grads=agg_grad, parameters=self.model.parameters())
        # apply gradient clipping
        if self.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.max_grad_norm)
        # do optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # update the model weights
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

    # ------------------------------------------------- #
    #               GAR Implementations                 #
    # ------------------------------------------------- #

    def __fed_avg(self, clients: List[Client]):
        """
        This implements classic FedAvg: McMahan et al., Communication-Efficient Learning of Deep
        Networks from Decentralized Data, NeuRips 2017
        :param clients: List of client nodes to aggregate over
        """
        # compute average grad
        agg_grad = self.weighted_average(clients=clients)
        return agg_grad

    def __fed_median(self, clients: List[Client]):
        """
        This is implementation of Geometric Median Aggregation proposed in :
        Pillutla et.al. Robust Aggregation for Federated Learning. 	arXiv:1912.13445
        """
        raise NotImplementedError

    def __m_krum(self, clients: List[Client], frac_m: float = 0.7):
        """
        This is an implementation of m-krum
        :param clients: List of all clients participating in training
        :param frac_m: m=n-f i.e. total-mal_nodes , since in practice server won't know this treat as hyper-param
        :return: List of clients that satisfies alpha-f byz resilience.
        """
        # compute mutual distance of each clients in terms of their grads
        dist = self.get_krum_dist(clients=clients)
        # compute estimated 'benign' client count / or num of closest nodes to consider
        m = frac_m * len(clients)
        # initialize min error to something large and min client ix
        min_error = 1e10
        min_err_client_ix = -1
        for client_ix in dist.keys():
            errors = sorted(dist[client_ix].values())
            curr_error = sum(errors[:m])
            if curr_error < min_error:
                min_error = curr_error
                min_err_client_ix = client_ix

        krum_grad = clients[min_err_client_ix].grad
        return krum_grad

    @staticmethod
    def weighted_average(clients, alphas=None):
        """
        Implements weighted average of client grads if no weights are supplied
        then its equivalent to simple average / Fed Avg
        """
        if alphas is None:
            alphas = [1] * len(clients)
        agg_grad = np.zeros_like(clients[0].grad)
        tot = np.sum(alphas)
        for alpha, client in zip(alphas, clients):
            agg_grad += (alpha / tot) * client.grad

        return agg_grad

    @staticmethod
    def get_krum_dist(clients) -> defaultdict:
        """ Computes distance between each pair of client based on grad value """
        dist = defaultdict(dict)
        for i in range(len(clients)):
            for j in range(i):
                dist[i][j] = dist[j][i] = np.linalg.norm(clients[i].grad - clients[j].grad)
        return dist



