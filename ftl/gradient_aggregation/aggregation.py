from ftl.models.model_helper import dist_weights_to_model, dist_grads_to_model
from ftl.training_utils.optimization import SchedulingOptimization
from ftl.agents.client import Client
from ftl.comm_compression.fast_lr_decomp import FastLRDecomposition
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
import copy


class Aggregator:
    """
    This class updates a global model with gradients aggregated from clients and
    keeps track of the model object and weights.
    """
    def __init__(self, agg_strategy: str,
                 model: nn.Module,
                 rank: int = None,
                 adaptive_k_th: float = None,
                 dual_opt_alg: str = "Adam",
                 opt_group: Dict = None,
                 max_grad_norm: float = None,
                 m_krum: float = 0.7):
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
        self.rank = rank
        self.adaptive_k_th = adaptive_k_th

        # Instantiate the optimizer for an aggregator
        server_opt = SchedulingOptimization(model=model, opt_alg=dual_opt_alg, opt_group=opt_group)
        self.optimizer = server_opt.optimizer
        self.lr_scheduler = server_opt.lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.krum_frac = m_krum
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.model.parameters()])

        # Bad Handling Fix with Base Class
        self.Sigma = []

    def set_lr(self, current_lr):
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr

    def update_model(self, clients: List[Client], current_lr: float = 0.01, alphas: np.ndarray = None) -> np.array:
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
        if self.agg_strategy == 'fed_avg':
            agg_grad = self.__fed_avg(G=G, alphas=alphas)
        elif self.agg_strategy == 'fed_lr_avg':
            agg_grad, Sigma = self.__fed_lr_avg(stacked_grad=G, k=self.rank,
                                                adaptive_k_th=self.adaptive_k_th)
            self.Sigma.append(Sigma)
        elif self.agg_strategy == 'krum':
            agg_grad, _ = self.__m_krum(clients=clients, frac_m=self.krum_frac)
        else:
            raise NotImplementedError

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

    # ------------------------------------------------- #
    #               GAR Implementations                 #
    # ------------------------------------------------- #
    # TODO:: move to Common Class
    def __fed_avg(self, G: np.ndarray, alphas: np.ndarray = None) -> np.ndarray:
        """
        This implements classic FedAvg: McMahan et al., Communication-Efficient Learning of Deep
        Networks from Decentralized Data, (NeuRips 2017)
        :return: aggregated gradient
        """

        agg_grad = self.weighted_average(stacked_grad=G, alphas=alphas)
        return agg_grad

    def __fed_lr_avg(self, stacked_grad: np.ndarray,
                     k: int,
                     adaptive_k_th: float) -> Tuple[np.ndarray, List[float]]:
        """
        Implements proposed Faster Convergence of FL through MF: Acharya. A.

        :param clients: List of client nodes to aggregate over
        :param k: perform (k) rank svd
        :return: aggregated gradient
        """
        if k is None:
            k = min(stacked_grad.shape[0], stacked_grad.shape[1])
        lr_factorization = FastLRDecomposition(n_components=k,
                                               X=stacked_grad,
                                               adaptive_k_th=adaptive_k_th)

        return lr_factorization.agg_grad, lr_factorization.Sigma

    def __fed_median(self, clients: List[Client]):
        """
        This is implementation of Geometric Median Aggregation proposed in :
        Pillutla et.al. Robust Aggregation for Federated Learning. 	arXiv:1912.13445
        """
        raise NotImplementedError

    def __m_krum(self, clients: List[Client], frac_m: float = 0.7) -> [np.ndarray, int]:
        """
        This is an implementation of m-krum
        :param clients: List of all clients participating in training
        :param frac_m: m=n-f i.e. total-mal_nodes , since in practice server won't know this treat as hyper-param
        :return: aggregated gradient, ix of worker selected by alpha-f Byz resilience
        """
        dist = self.get_krum_dist(clients=clients)
        m = int(frac_m * len(clients))
        min_score = 1e10
        optimal_client_ix = -1
        for ix in range(len(clients)):
            curr_dist = dist[ix, :]
            curr_dist = np.sort(curr_dist)
            curr_score = sum(curr_dist[:m])
            if curr_score < min_score:
                min_score = curr_score
                optimal_client_ix = ix
        krum_grad = clients[optimal_client_ix].grad
        print('Krum picked client {}'.format(clients[optimal_client_ix].client_id))

        return krum_grad, optimal_client_ix

    def bulyan(self, clients: List[Client], frac_m: float = 0.7):
        raise NotImplementedError

    @staticmethod
    def weighted_average(stacked_grad: np.ndarray, alphas=None):
        """
        Implements weighted average of client grads i.e. rows of G
        If no weights are supplied then its equivalent to simple average / Fed Avg
        """
        if alphas is None:
            alphas = [1.0 / stacked_grad.shape[0]] * stacked_grad.shape[0]
        else:
            assert len(alphas) == stacked_grad.shape[0]
        agg_grad = np.zeros_like(stacked_grad[0, :])
        for ix in range(0, stacked_grad.shape[0]):
            agg_grad += alphas[ix] * stacked_grad[ix, :]
        return agg_grad

    @staticmethod
    def get_krum_dist(clients) -> np.ndarray:
        """ Computes distance between each pair of client based on grad value """
        dist = np.zeros((len(clients), len(clients)))
        for i in range(len(clients)):
            for j in range(i):
                dist[i][j] = dist[j][i] = np.linalg.norm(clients[i].grad - clients[j].grad)
        return dist

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
