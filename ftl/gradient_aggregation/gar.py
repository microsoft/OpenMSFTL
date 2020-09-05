# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

import numpy as np
from typing import List, Dict
import torch
from ftl.gradient_aggregation.spectral_aggregation import RobustPCAEstimator, fast_lr_decomposition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAR:
    """
    This is the base class for all the implement GAR
    """

    def __init__(self, aggregation_config: Dict):
        self.aggregation_config = aggregation_config
        self.gradient_weights = None  # weight on gradient on each client
        self.Sigma_tracked = []
        self.alpha_tracked = []

    def aggregate(self, G: np.ndarray,
                  client_ids: np.ndarray,
                  losses: List[float]) -> np.ndarray:
        pass

    def weighted_average(self, stacked_grad: np.ndarray):
        """
        Implements weighted average of client grads i.e. rows of G
        If no weights are supplied then its equivalent to simple average / Fed Avg
        """
        if self.gradient_weights is None:
            self.gradient_weights = np.full(stacked_grad.shape[0],
                                            fill_value=1.0 / stacked_grad.shape[0],
                                            dtype=stacked_grad.dtype)
        else:
            assert len(self.gradient_weights) == stacked_grad.shape[0]
        # row-wise multiplication with the weight vector and sum the weighted row vectors
        agg_grad = np.sum(np.multiply(stacked_grad, self.gradient_weights[:, np.newaxis]), axis=0)

        return agg_grad


class FedAvg(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray,
                  client_ids: np.ndarray = None,
                  losses: List[float] = None) -> np.ndarray:
        agg_grad = self.weighted_average(stacked_grad=G)
        return agg_grad


class MinLoss(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray,
                  losses: List[float],
                  client_ids: np.ndarray = None) -> np.ndarray:
        if not losses:
            raise Exception("To use MinLoss GAR , you must provide losses to aggregate call")
        min_loss_ix = losses.index(min(losses))
        return G[min_loss_ix, :]


class SpectralFedAvg(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)
        self.rank = self.aggregation_config["rank"]
        self.adaptive_rank_th = self.aggregation_config["adaptive_rank_th"]
        self.drop_top_comp = self.aggregation_config["drop_top_comp"]
        self.num_clients = self.aggregation_config["num_client_nodes"]
        self.auto_encoder_init_steps = self.aggregation_config.get("num_encoder_init_epochs", 2000)
        self.auto_encoder_fine_tune_steps = self.aggregation_config.get("num_encoder_ft_epochs", 1000)
        self.pca = None
        self.analytic = self.aggregation_config.get("analytic", False)
        self.auto_encoder_loss = self.aggregation_config.get("auto_encoder_loss", "scaled_mse")

    def aggregate(self, G: np.ndarray,
                  client_ids: np.ndarray,
                  losses:  List[float] = None) -> np.ndarray:
        if self.analytic:
            # Perform Analytic Randomized PCA
            G_approx, S = fast_lr_decomposition(X=G,
                                                rank=self.rank,
                                                adaptive_rank_th=self.adaptive_rank_th,
                                                drop_top_comp=self.drop_top_comp)
            self.Sigma_tracked.append(S)
            agg_grad = self.weighted_average(stacked_grad=G_approx)
            return agg_grad

        else:
            # Else: we train a linear auto-encoder
            G = torch.from_numpy(G).to(device)
            if self.pca is None:
                self.pca = RobustPCAEstimator(self.num_clients, G.shape[1], self.rank, device,
                                              auto_encoder_loss=self.auto_encoder_loss)
                self.pca.fit(G, client_ids, steps=self.auto_encoder_init_steps)
            else:
                self.pca.fine_tune(G, client_ids, steps=self.auto_encoder_fine_tune_steps)

            G_approx, scales = self.pca.transform(G, client_ids)
            self.alpha_tracked.append(scales)
            cut = 1 - self.adaptive_rank_th
            print("cutting off {}% of components".format(cut*100))
            k = int(np.ceil(cut * scales.shape[0]))
            cutoff = torch.min(torch.topk(scales, k=k)[0])
            alphas = torch.ones_like(scales) * (scales < cutoff).to(torch.float32)
            return torch.einsum("nf,n->f", G_approx, alphas).detach().cpu().numpy()


class Krum(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray,
                  client_ids: np.ndarray = None,
                  losses: List[float] = None) -> np.ndarray:
        dist = self.get_krum_dist(G=G)
        m = int(self.aggregation_config.get("krum_frac", 0.3) * G.shape[0])
        min_score = 1e10
        optimal_client_ix = -1

        for ix in range(G.shape[0]):
            curr_dist = dist[ix, :]
            curr_dist = np.sort(curr_dist)
            curr_score = sum(curr_dist[:m])
            if curr_score < min_score:
                min_score = curr_score
                optimal_client_ix = ix
        krum_grad = G[optimal_client_ix, :]
        return krum_grad

    @staticmethod
    def get_krum_dist(G: np.ndarray) -> np.ndarray:
        """ Computes distance between each pair of client based on grad value """
        dist = np.zeros(G.shape[0], G.shape[0])  # num_clients * num_clients
        for i in range(G.shape[0]):
            for j in range(i):
                dist[i][j] = dist[j][i] = np.linalg.norm(G[i, :] - G[j, :])
        return dist
