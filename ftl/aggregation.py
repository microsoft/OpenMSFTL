import numpy as np
import torch.nn as nn
from ftl.models import dist_weights_to_model, add_grads_to_model
from ftl.optimization import SchedulingOptimization


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object and weights.

    This implements the following aggregation algorithms:
    1. FedAvg :
    -------------------
        a.  Simple FedAvg aggregation as introduced in:
            McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data",
            http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
        b.  Federated dual optimization described in:
            D. Dimitriadis et al., "On a Federated Approach for Training Acoustic Models ", Interspeech 2021,
            S. J. Reddi et al., "Adaptive Federated Optimization", arXiv:2003.00295
    """

    def __init__(self, agg_strategy, model, opt_alg=None, opt_group={}, max_grad_norm=None):
        """
        :param agg_strategy: aggregation strategy, default to "fed_avg"
        :type agg_strategy: str
        :param model: class:`nn.Module`, the global model
        :type model: subclass of class:`nn.Module`
        :param opt_alg: type of (adaptive) federated optimizater; see exmaples for ftl/optimization.py
        :type opt_alg: string
        :param opt_group: parameters for the optimizer; see details for ftl/optimization.py
        :type opt_group: dict
        :param max_grad_norm: max norm of the gradients for gradient clipping, default to None
        :type max_grad_norm: float
        """
        self.agg_strategy = agg_strategy
        self.model = model
        # Instantiate the optimizer for an aggregator
        if opt_alg is None or opt_alg == "None":  # simple model averaging
            self.optimizer = None
            self.lr_scheduler = None
        else:  # dual optimization
            sopt = SchedulingOptimization(model=model,
                                          opt_alg=opt_alg,
                                          opt_group=opt_group)
            self.optimizer = sopt.optimizer
            self.lr_scheduler = sopt.lr_scheduler

        self.max_grad_norm = max_grad_norm
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

    def set_lr(self, current_lr):
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

    def update_model(self, clients, current_lr=None):
        """
        Update a model with aggregated gradients

        :param current_lr:
        :param clients: A set of client compute nodes
        :type clients: list of class:`ftl.Client`
        :return: The weights of the updated global model
        """

        if self.agg_strategy is 'fed_avg':
            self.fed_avg(clients=clients, current_lr=current_lr)
            # update the model params with these weights
            dist_weights_to_model(weights=self.w_current, parameters=self.model.parameters())
        else:
            raise NotImplementedError
        return self.w_current

    def fed_avg(self, clients, current_lr: float):
        if self.optimizer is None:
            """ perform the Vanilla FedAvg """
            client_grads = np.empty((len(clients), len(self.w_current)), dtype=self.w_current.dtype)
            for ix, client in enumerate(clients):
                client_grads[ix, :] = client.grad

            lr = current_lr if current_lr is not None else 1.0
            self.w_current -= lr * np.mean(client_grads, axis=0)

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


