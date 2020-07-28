from ftl.models.model_helper import dist_weights_to_model, dist_grads_to_model
from ftl.training_utils.optimization import SchedulingOptimization
from ftl.agents.client import Client
from .gar import FedAvg, SpectralFedAvg
import torch.nn as nn
from typing import Dict, List
import numpy as np
import copy


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object and weights.
    """
    def __init__(self,
                 aggregation_config: Dict,
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
        self.aggregation_config = aggregation_config
        self.gar = self.get_gar()
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

    def get_gar(self):
        if self.aggregation_config["aggregation_scheme"] == 'fed_avg':
            return FedAvg(aggregation_config=self.aggregation_config)
        elif self.aggregation_config["aggregation_scheme"] == 'fed_lr_avg':
            return SpectralFedAvg(aggregation_config=self.aggregation_config)
        else:
            raise NotImplementedError

    def update_model(self, clients: List[Client],
                     current_lr: float = 0.01,
                     alphas: np.ndarray = None) -> np.array:
        """
        Update server model with aggregated gradients
        :param current_lr:
        :param clients: A set of client compute nodes
        :param alphas: Weight for each gradient
        :return: The weights of the updated global model
        """
        if len(clients) == 0:
            raise Exception('Client List is Empty')
        # create a stacked Gradient Matrix G = [G0 | G1 | .... | Gn]'
        # each row corresponds to gradient vector for each client
        G = np.zeros((len(clients), len(clients[0].grad)), dtype=clients[0].grad.dtype)
        for ix, client in enumerate(clients):
            G[ix, :] = client.grad
        agg_grad = self.gar.aggregate(G=G, alphas=alphas)

        # Now Run a step of Dual Optimizer (Server Step)
        self.__server_step(agg_grad=agg_grad, current_lr=current_lr)
        dist_weights_to_model(weights=self.w_current, parameters=self.model.to('cpu').parameters())
        return self.w_current

    def __server_step(self, agg_grad, current_lr: float):
        """
        Implements Server (Dual) Optimization with the option to use Adaptive Optimization
        D. Dimitriadis et al., "On a Federated Approach for Training Acoustic Models ", Interspeech 2021,
        S. J. Reddi et al., "Adaptive Federated Optimization", arXiv:2003.00295
        """
        self.set_lr(current_lr)
        # Update the model with aggregated gradient
        dist_grads_to_model(grads=agg_grad, parameters=self.model.to('cpu').parameters())
        # apply gradient clipping
        if self.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 self.max_grad_norm)
        # do optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # update the model weights
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

    #####################
    def state_dict(self):
        """Returns the state of the aggregator as a :class:`dict`.

        It contains four entries:
        * model_state_dict - a dict with the model state.
        * w_current - a dict holding the model weights.
        * optimizer_state_dict - a dict containing the optimizer state.
        * lr_scheduler_state_dict - a dict keeping the LR scheduler state.
        """
        return {
            'model_state_dict': self.model.state_dict(),
            'w_current': copy.deepcopy(self.w_current),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }

    def load_state_dict(self, state_dict):
        """Loads the aggregator state.

        param state_dict: aggregator state. Should be an object returned from a call to :meth:`state_dict`.
        """
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.w_current = state_dict['w_current']
        dist_weights_to_model(weights=self.w_current, parameters=self.model.to('cpu').parameters())

        if state_dict['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

        if state_dict['lr_scheduler_state_dict'] is not None:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler_state_dict'])
