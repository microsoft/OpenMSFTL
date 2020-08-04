import numpy as np
from sklearn.utils.extmath import randomized_svd


class GAR:
    """
    This is the base class for all the implement GAR
    """
    def __init__(self, aggregation_config):
        self.aggregation_config = aggregation_config
        self.Sigma_tracked = []

    def aggregate(self, G: np.ndarray) -> np.ndarray:
        pass

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


class FedAvg(GAR):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)

    def aggregate(self, G: np.ndarray) -> np.ndarray:
        agg_grad = self.weighted_average(stacked_grad=G, alphas=None)
        return agg_grad


class SpectralFedAvg(FedAvg):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)
        self.rank = self.aggregation_config["rank"]
        self.adaptive_rank_th = self.aggregation_config["adaptive_rank_th"]
        self.drop_top_comp = self.aggregation_config["drop_top_comp"]

    def aggregate(self, G: np.ndarray) -> np.ndarray:
        G_approx, S = self.fast_lr_decomposition(X=G)
        self.Sigma_tracked.append(S)
        agg_grad = self.weighted_average(stacked_grad=G_approx, alphas=None)
        return agg_grad

    def fast_lr_decomposition(self, X):
        if not self.rank:
            self.rank = min(X.shape[0], X.shape[1])
        print('Doing a {} rank SVD'.format(self.rank))
        X = np.transpose(X)
        U, S, V = randomized_svd(X, n_components=self.rank,
                                 flip_sign=True)
        if self.adaptive_rank_th:
            if not 0 < self.adaptive_rank_th < 1:
                raise Exception('adaptive_rank_th should be between 0 and 1')
            n_samples, n_features = X.shape
            explained_variance_ = (S ** 2) / (n_samples - 1)
            total_var = np.var(X, ddof=1, axis=0)
            explained_variance_ratio_ = explained_variance_ / total_var.sum()
            # print(explained_variance_ratio_)
            cum_var_explained = np.cumsum(explained_variance_ratio_)
            print(cum_var_explained)
            adaptive_rank = np.searchsorted(cum_var_explained, v=self.adaptive_rank_th)
            print('Truncating Spectral Grad Matrix to rank {} using '
                  '{} threshold'.format(adaptive_rank, self.adaptive_rank_th))
            U_k = U[:, 0:adaptive_rank]
            S_k = S[0:adaptive_rank]
            V_k = V[0:adaptive_rank, :]

        else:
            U_k = U
            S_k = S
            V_k = V

        if self.drop_top_comp:
            U_k = U_k[:, 1:]
            S_k = S_k[1:]
            V_k = V_k[1:, :]

        lr_approx = np.dot(U_k * S_k, V_k)
        lr_approx = np.transpose(lr_approx)
        return lr_approx, S

    # def __m_krum(self, clients: List[Client], frac_m: float = 0.7) -> [np.ndarray, int]:
    #     """
    #     This is an implementation of m-krum
    #     :param clients: List of all clients participating in training
    #     :param frac_m: m=n-f i.e. total-mal_nodes , since in practice server won't know this treat as hyper-param
    #     :return: aggregated gradient, ix of worker selected by alpha-f Byz resilience
    #     """
    #     dist = self.get_krum_dist(clients=clients)
    #     m = int(frac_m * len(clients))
    #     min_score = 1e10
    #     optimal_client_ix = -1
    #     for ix in range(len(clients)):
    #         curr_dist = dist[ix, :]
    #         curr_dist = np.sort(curr_dist)
    #         curr_score = sum(curr_dist[:m])
    #         if curr_score < min_score:
    #             min_score = curr_score
    #             optimal_client_ix = ix
    #     krum_grad = clients[optimal_client_ix].grad
    #     print('Krum picked client {}'.format(clients[optimal_client_ix].client_id))
    #
    #     return krum_grad, optimal_client_ix
    #
    # @staticmethod
    # def get_krum_dist(clients) -> np.ndarray:
    #     """ Computes distance between each pair of client based on grad value """
    #     dist = np.zeros((len(clients), len(clients)))
    #     for i in range(len(clients)):
    #         for j in range(i):
    #             dist[i][j] = dist[j][i] = np.linalg.norm(clients[i].grad - clients[j].grad)
    #     return dist


