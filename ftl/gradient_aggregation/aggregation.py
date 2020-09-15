# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict, List
import copy
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from ftl.agents.client import Client
from ftl.training_utils import infer
from ftl.models.model_helper import dist_weights_to_model, dist_grads_to_model
from ftl.gradient_aggregation.weight_estimator import RLWeightEstimator, SoftmaxLWeightEstimator
from .gar import FedAvg, SpectralFedAvg
from sklearn.utils.extmath import randomized_svd


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object with weights, optimizer and LR scheduler.
    """

    def __init__(self, aggregation_config: Dict,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       clip_val: float,
                       lr_scheduler: optim.lr_scheduler._LRScheduler):
        self.aggregation_config = aggregation_config
        self.model = model
        self.opt = optimizer
        self.clip_val = clip_val if isinstance(clip_val, float) is True else -1.0
        self.lrs = lr_scheduler
        self.gar = self.__get_gar()
        self.curr_G = None
        self.agg_grad = None
        self.analyze_pc = self.aggregation_config.get("pc_analysis", False)

    def __get_gar(self):
        """
        Wrapper function to select an appropriate aggregation function
        """
        if self.aggregation_config["aggregation_scheme"] == 'fed_avg':
            return FedAvg(aggregation_config=self.aggregation_config)
        elif self.aggregation_config["aggregation_scheme"] == 'fed_spectral_avg':
            return SpectralFedAvg(aggregation_config=self.aggregation_config)
        else:
            raise NotImplementedError

    def aggregate_grads(self, clients: List[Client], input_feature: np.ndarray=None, val_loader: DataLoader=None) -> np.array:
        """
        Aggregate gradient information from clients
        """
        if len(clients) == 0:
            raise Exception('Client List is Empty')
        G = np.zeros((len(clients), len(clients[0].grad)), dtype=clients[0].grad.dtype)
        for ix, client in enumerate(clients):
            G[ix, :] = client.C.compress(client.grad)
        if self.analyze_pc is True:
            print("Randomized SVD of G for Spectral Analysis")
            U, S, V = randomized_svd(G, n_components=min(G.shape[0], G.shape[1]), flip_sign=True)
            self.gar.Sigma_tracked.append(S)
        self.curr_G = G
        client_ids = np.array([c.client_id for c in clients])
        self.agg_grad = self.gar.aggregate(G=G, client_ids=client_ids)

    def update_model(self):
        """
        Update the model with aggregated gradient
        """
        dist_grads_to_model(grads=np.array(self.agg_grad, dtype=np.float32), parameters=self.model.to('cpu').parameters())
        if self.clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_val, norm_type=1)
        self.opt.step()
        if self.lrs:
            print('Aggregator lr = {}'.format(self.lrs.get_lr()))
            self.lrs.step()

        model_weights = np.concatenate([w.data.numpy().flatten() for w in self.model.to('cpu').parameters()])
        dist_weights_to_model(weights=model_weights, parameters=self.model.to('cpu').parameters())

        return model_weights

    def state_dict(self):
        """
        Returns the state of the aggregator as a :class:`dict`.

        It contains four entries:
        * model_state_dict - a dict with the model state.
        * optimizer_state_dict - a dict containing the optimizer state.
        * lr_scheduler_state_dict - a dict keeping the LR scheduler state.
        """
        return {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.opt.state_dict() if self.opt is not None else None,
            'lr_scheduler_state_dict' : self.lrs.state_dict() if self.lrs is not None else None
        }

    def load_state_dict(self, state_dict):
        """
        Loads the aggregator state.
        """
        self.model.load_state_dict(state_dict['model_state_dict'])
        if state_dict['optimizer_state_dict'] is not None:
           self.opt.load_state_dict(state_dict['optimizer_state_dict'])

        if state_dict['lr_scheduler_state_dict'] is not None:
            self.lrs.load_state_dict(state_dict['lr_scheduler_state_dict'])


class DGAggregator(Aggregator):
    """
    This implements a dynamic gradient aggregator that linearly combines gradient vectors
    """
    def __init__(self, aggregation_config: Dict,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       clip_val: float,
                       lr_scheduler: optim.lr_scheduler._LRScheduler):
        super(DGAggregator, self).__init__(aggregation_config=aggregation_config,
                                            model=model,
                                            optimizer=optimizer,
                                            clip_val=clip_val,
                                            lr_scheduler=lr_scheduler)

        # set a weight estimator for each client gradient
        self.weight_estimator = None
        self.dga_config = aggregation_config['dga_config']
        dga_type = self.dga_config['type']
        if dga_type == 'RL':
            self.weight_estimator = RLWeightEstimator(self.dga_config)
        elif dga_type == 'softmax':
            self.weight_estimator = SoftmaxLWeightEstimator(self.dga_config)
        else:
            raise ValueError("Invalid gradient estimator type {}".format(dga_type))

        # This will be used in contrast to the RL-based method
        self.sub_weight_estimator = None
        if self.dga_config.get('subtype', '') == 'softmax':
            print("Additional weight estimator: {}".format(self.dga_config['subtype']))
            self.sub_weight_estimator = SoftmaxLWeightEstimator(self.dga_config['sub_dga_config'])
        elif 'subtype' in self.dga_config:
            raise ValueError("Invalid gradient estimator type: 'subtype': {}".format(self.dga_config['subtype']))


    def aggregate_grads(self, clients: List[Client],
                              input_feature: np.ndarray,
                              val_loader: DataLoader):
        """
        Perform dynamic aggregation with a selected method: softmax, RL or RL with softmax
        """
        if self.weight_estimator.estimator_type == 'softmax':
            self.gar.gradient_weights = self.weight_estimator.compute_weights(input_feature, len(clients))
            Aggregator.aggregate_grads(self, clients=clients)
            self.update_model()
        elif self.weight_estimator.estimator_type == 'RL':
            org_aggregator_state = copy.deepcopy(self.state_dict())
            # run RL-based weight estimator and update a model with weighted aggregation
            self.gar.gradient_weights = self.weight_estimator.compute_weights(input_feature)  # set the weight vector for each gradient
            Aggregator.aggregate_grads(self, clients=clients)
            self.update_model()
            rl_aggregator_state = self.state_dict()
            val_wi_rl = infer(test_loader=val_loader, model=self.model)
            print('Acc with RL: {}'.format(val_wi_rl))

            # update a model without RL
            if self.sub_weight_estimator is not None:
                # perform softmax-weighting
                self.load_state_dict(org_aggregator_state)  # revert to the original state
                self.gar.gradient_weights = self.sub_weight_estimator.compute_weights(input_feature, len(clients))
                Aggregator.aggregate_grads(self, clients=clients)
                self.update_model()
                val_wo_rl = infer(test_loader=val_loader, model=self.model)
                print('Acc with Softmax DGA: {}'.format(val_wo_rl))
            else:
                # aggregate without the weight
                self.load_state_dict(org_aggregator_state)  # revert to the original state
                self.gar.gradient_weights = None  # will use a uniform weight
                Aggregator.aggregate_grads(self, clients=clients)
                self.update_model()
                val_wo_rl = infer(test_loader=val_loader, model=self.model)
                print('Acc wo DGA: {}'.format(val_wo_rl))

            # update the RL model
            should_use_rl_model = self.weight_estimator.update_model(input_feature, 1 - val_wi_rl / 100.0, 1 - val_wo_rl / 100.0)
            if should_use_rl_model is True:  # keep the model updated with RL-based weighted aggregation
                self.load_state_dict(rl_aggregator_state)
        else:
            raise ValueError("Invalid weight estimation method: {}".format(self.weight_estimator.estimator_type))


def make_aggregator(aggregator_config: Dict,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    clip_val: float,
                    lr_scheduler: optim.lr_scheduler._LRScheduler):
    """
    Create an aggregator instance
    """

    if aggregator_config.get('dga_config', None) is not None:
        print("Dynamic Gradient Aggregator", flush=True)
        aggregator = DGAggregator(aggregation_config=aggregator_config,
                                model=model,
                                optimizer=optimizer,
                                clip_val=clip_val,
                                lr_scheduler=lr_scheduler)
    else:
        print("Conventional Gradient Aggregator", flush=True)
        aggregator = Aggregator(aggregation_config=aggregator_config,
                                model=model,
                                optimizer=optimizer,
                                clip_val=clip_val,
                                lr_scheduler=lr_scheduler)

    return aggregator