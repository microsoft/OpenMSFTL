from ftl.models import dist_weights_to_model, dist_grads_to_model
from ftl.optimization import SchedulingOptimization
from ftl.client import Client
from ftl.agg_utils import weighted_average
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

    def update_model(self, clients: List[Client], current_lr: float = None) -> np.array:
        """
        Update a model with aggregated gradients
        :param current_lr:
        :param clients: A set of client compute nodes
        :return: The weights of the updated global model
        """
        if self.agg_strategy is 'fed_avg':
            self.__fed_avg(clients=clients, current_lr=current_lr)
            dist_weights_to_model(weights=self.w_current, parameters=self.model.parameters())
        else:
            raise NotImplementedError
        return self.w_current

    def __fed_avg(self, clients: List[Client], current_lr: float = 0.01):
        """
        This implements two flavors the Federated Averaging GAR:
            a.  Simple FedAvg aggregation as introduced in:
                McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data",
                http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
            b.  FedAvg with dual optimization described in:
                D. Dimitriadis et al., "On a Federated Approach for Training Acoustic Models ", Interspeech 2021,
                S. J. Reddi et al., "Adaptive Federated Optimization", arXiv:2003.00295

        :param clients: List of client nodes to aggregate over
        :param current_lr: supply the current lr for the Update step
        """
        # compute average grad
        agg_grad = weighted_average(clients=clients)
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

        # get the model weights
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

    def __fed_median(self, clients: List[Client]):
        """
        This is implementation of Geometric Median Aggregation proposed in :
        Pillutla et.al. Robust Aggregation for Federated Learning. 	arXiv:1912.13445
        """
        raise NotImplementedError

    def __krum(self):
        raise NotImplementedError


