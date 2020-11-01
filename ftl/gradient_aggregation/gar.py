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
        self.num_updates = 0

    def aggregate(self, G: np.ndarray,
                  client_ids: np.ndarray) -> np.ndarray:
        """
        Method that implements gradient aggregation
        :param G: M x N matrix where M is no. clients and N is the dimension of the gradient vector
        """
        self.num_updates += 1
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
                  client_ids: np.ndarray = None) -> np.ndarray:
        agg_grad = self.weighted_average(stacked_grad=G)
        return agg_grad


class SpectralFedAvg(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)
        self.num_clients = self.aggregation_config.get("num_sampled_clients",
                                                       self.aggregation_config["num_client_nodes"])
        self.rank = self.aggregation_config["rank"]
        self.adaptive_rank_th = self.aggregation_config.get("adaptive_rank_th", -1)
        assert self.adaptive_rank_th <= 1.0, "Invalid 'adaptive_rank_th': {} > 1".format(self.adaptive_rank_th)
        # how do we create a subspace, filtering malicious clients or remove noisy gradient components?
        # 0 if it is client filtering and 1 if dimensionality reduction
        self.subspace_axis = self.aggregation_config.get("subspace_axis", 1)
        self.analytic = self.aggregation_config.get("analytic", False)
        if self.analytic is True:
            self.drop_top_comp = self.aggregation_config["drop_top_comp"]
        else:
            self.auto_encoder_init_steps = self.aggregation_config.get("num_encoder_init_epochs", 2000)
            self.auto_encoder_fine_tune_steps = self.aggregation_config.get("num_encoder_ft_epochs", 1000)
            self.auto_encoder_loss = self.aggregation_config.get("auto_encoder_loss", "scaled_mse")
            self.auto_encoder_fit_freq = self.aggregation_config.get("auto_encoder_fit_freq", 64)  # how frequently a fitting process (full training) is peformanced

        self.pca = None
        self.cov_mats = None  # Covariance matrix for each base: torch.zeros(self.rank, self.num_clients, self.num_clients)
        self.forgetting_factor = self.aggregation_config.get("forgetting_factor", 0.9)
        self.diagonal_loading = self.aggregation_config.get("diagonal_loading", 0.01)
        self.num_noisy_clients = self.aggregation_config.get("num_noisy_clients", self.num_clients // 10)

    def aggregate(self, G: np.ndarray,
                  client_ids: np.ndarray) -> np.ndarray:

        if self.analytic is True:
            # Perform Analytic Randomized PCA
            G_updated = self.conventional_pca_aggregation(G)
        else:
            # Train a linear auto-encoder
            if self.subspace_axis == 2:
                (G_approx, scales) = self.mk_pca_space_filtering(G, client_ids)
            elif self.subspace_axis == 1:
                (G_approx, scales) = self.mk_pca_dimensionality_reduction(G, client_ids)
            elif self.subspace_axis == 0:
                (G_approx, scales) = self.mk_pca_client_filtering(G)
            else:
                raise NotImplementedError("Invalid subspace axis {}".format(self.subspace_axis))

            self.alpha_tracked.append(scales)

            def _lre_rank_selection(scales, adaptive_rank_th):
                """
                Filter data samples with a larger log-RMSE.
                notes:
                This is implemented by A. Acharya and left for maintaining backword compatibility
                """
                cut = 1 - adaptive_rank_th
                k = int(np.ceil(cut * scales.shape[0]))
                cutoff = torch.min(torch.topk(scales, k=k)[0])
                print("cutting off {}% of components: {} out of {}: with logstd threshold={}".format(cut*100, k, scales.shape[0], cutoff))
                return torch.ones_like(scales) * (scales < cutoff).to(torch.float32)

            if self.adaptive_rank_th >= 0.0:
                alphas = _lre_rank_selection(scales, self.adaptive_rank_th)
                if self.subspace_axis == 1:
                    G_updated = torch.einsum("nf,n->f", G_approx, alphas)
                elif self.subspace_axis == 2:
                    G_approx = torch.einsum("nf,n->nf", G_approx, alphas)
                else:  # row-wise multiplication and sum along the column (client axis=0)
                    G_updated = torch.einsum("nf,f->f", G_approx, alphas)
            else:  # No selection is performed
                G_updated = torch.sum(G_approx, 0)

        self.num_updates += 1
        return G_updated.detach().cpu().numpy()

    def conventional_pca_aggregation(self, G: np.ndarray) -> np.ndarray:
        """
        Run subspace filtering with conventional SVD-based PCA
        """
        print("Conventional PCA...")
        G_approx, S = fast_lr_decomposition(X=G,
                                            rank=self.rank,
                                            adaptive_rank_th=self.adaptive_rank_th,
                                            drop_top_comp=self.drop_top_comp)
        self.Sigma_tracked.append(S)
        agg_grad = self.weighted_average(stacked_grad=G_approx)
        return agg_grad

    def mk_pca_space_filtering(self, G, client_ids):
        """
        Perform subspace filtering on the Mark Hamiliton PCA domain,
        where components associated with smaller eigenvalues are filtered

        :return: Tuple of gradient matrix and reconstruction error vector for each client
        """
        print("Subspace filtering on the Mark Hamilton PCA domain...")
        G = torch.from_numpy(G).to(device)
        if self.pca is None:
            self.pca = RobustPCAEstimator(self.num_clients, input_dim=G.shape[1], hidden_dim=self.rank, device=device,
                                        auto_encoder_loss=self.auto_encoder_loss)
            self.pca.fit(G, client_ids, steps=self.auto_encoder_init_steps)
        elif (self.num_updates % self.auto_encoder_fit_freq) != 0:  # update an auto-encoder only (in original Anish's implementation, it is always False)
            print("Fine-tuning")
            self.pca.fine_tune(G, client_ids, steps=self.auto_encoder_fine_tune_steps)
        else:  # also update a reconstruction error score
            print("Fitting at the {}-th update".format(self.num_updates))
            self.pca.fit(G, client_ids, steps=self.auto_encoder_init_steps)

        def _subspace_filtering(G, k):
            """
            Compute an eigenvector on a covariance matrix
            :param G:
            :param k: number of noisy clients (== no. eigenvector to ignore)
            """
            cov_mats = torch.einsum('ik, jk->kij', G, G)  # (#components, #cleints, #clients)
            if self.cov_mats is None: # compute an initial covariance matrix for each component
                self.cov_mats = cov_mats
            else:
                self.cov_mats = self.forgetting_factor * self.cov_mats + (1 - self.forgetting_factor) * cov_mats

            print("Cutting {} eigenvectors out of {}".format(k, G.shape[0]))
            es, Vs = torch.symeig(self.cov_mats + self.diagonal_loading * torch.eye(G.shape[0]), eigenvectors=True)
            # Perform subspace filtering with eigenvectors
            for i, v in enumerate(Vs):  # loop for components
                G[:,i] = torch.matmul(torch.matmul(v[:,k:], torch.transpose(v[:,k:], 0, 1)), G[:,i])
                #print("{}th component: ev th={}".format(i, es[i][k]))

            return G

        G_subspace, _ = self.pca.encode(G, None)
        if self.num_noisy_clients > 0:
            G_subspace = _subspace_filtering(G_subspace, self.num_noisy_clients)

        G_approx, scales = self.pca.decode(G_subspace, client_ids)
        return (G_approx, scales)

    def mk_pca_dimensionality_reduction(self, G, client_ids):
        """
        Use Mark Hamiliton PCA to reduce the dimension of a gradient vector
        and compute a reconstruction error score for each client

        :return: Tuple of gradient matrix and reconstruction error vector for each client
        """
        print("Mark Hamilton PCA for dimensionality reduction...")
        G = torch.from_numpy(G).to(device)
        if self.pca is None:
            self.pca = RobustPCAEstimator(self.num_clients, input_dim=G.shape[1], hidden_dim=self.rank, device=device,
                                        auto_encoder_loss=self.auto_encoder_loss)
            self.pca.fit(G, client_ids, steps=self.auto_encoder_init_steps)
        elif (self.num_updates % self.auto_encoder_fit_freq) != 0:  # update an auto-encoder only (in original Anish's implementation, it is always False)
            print("Fine-tuning")
            self.pca.fine_tune(G, client_ids, steps=self.auto_encoder_fine_tune_steps)
        else:  # also update a reconstruction error score
            print("Fitting at the {}-th update".format(self.num_updates))
            self.pca.fit(G, client_ids, steps=self.auto_encoder_init_steps)

        G_approx, scales = self.pca.transform(G, client_ids)
        return (G_approx, scales)

    def mk_pca_client_filtering(self, G):
        """
        Use Mark Hamiliton PCA to filter noisy clients
        and compute a reconstruction error score for each gradient vector component

        :return: Tuple of gradient matrix an reconstruction error vector for each gradient vector component
        """
        print("Mark Hamilton PCA for client filtering...")
        dataid = np.arange(G.shape[1])
        GT = torch.from_numpy(G.T).to(device)
        if self.pca is None:
            self.pca = RobustPCAEstimator(G.shape[1], input_dim=G.shape[0], hidden_dim=self.rank, device=device,
                                        auto_encoder_loss=self.auto_encoder_loss)
            self.pca.fit(GT, dataid, steps=self.auto_encoder_init_steps)
        elif (self.num_updates % self.auto_encoder_fit_freq) != 0:  # update an auto-encoder only
            print("Fine-tuning")
            self.pca.fine_tune(GT, dataid, steps=self.auto_encoder_fine_tune_steps)
        else:  # also update a reconstruction error score
            print("Fitting at the {}-th update".format(self.num_updates))
            self.pca.fit(GT, dataid, steps=self.auto_encoder_init_steps)

        GT_approx, scales = self.pca.transform(GT, dataid)
        return (torch.transpose(GT_approx, 0, 1), scales)


class Krum(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray,
                  client_ids: np.ndarray = None) -> np.ndarray:
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
