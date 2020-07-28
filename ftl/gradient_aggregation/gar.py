import numpy as np
from sklearn.utils.extmath import randomized_svd


class GAR:
    """
    This is the base class for all the implement GAR
    """
    def __init__(self, aggregation_config):
        self.aggregation_config = aggregation_config

    def aggregate(self, G: np.ndarray, alphas=None) -> np.ndarray:
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

    def aggregate(self, G: np.ndarray, alphas=None) -> np.ndarray:
        agg_grad = self.weighted_average(stacked_grad=G, alphas=alphas)
        return agg_grad


class SpectralFedAvg(FedAvg):
    def __init__(self, aggregation_config):
        GAR.__init__(self, aggregation_config=aggregation_config)
        self.rank = self.aggregation_config["rank"]
        self.adaptive_rank_th = self.aggregation_config["adaptive_k_th"]
        self.Sigma = None
        self.normalized_Sigma = None

    def aggregate(self, G: np.ndarray, alphas=None) -> np.ndarray:
        G_approx = self.fast_lr_decomposition(X=G)
        agg_grad = self.weighted_average(stacked_grad=G_approx, alphas=alphas)
        return agg_grad

    def fast_lr_decomposition(self, X):
        print('Doing a {} rank SVD'.format(self.rank))
        U, S, V = randomized_svd(X, n_components=self.rank,
                                 n_iter='auto',
                                 flip_sign=True)
        self.Sigma = S
        self.normalized_Sigma = S / sum(S)
        if 0 < self.adaptive_rank_th < 1:
            running_pow = 0.0
            adaptive_rank = 0
            for sv in self.normalized_Sigma:
                running_pow += sv
                adaptive_rank += 1
                if running_pow >= self.adaptive_rank_th:
                    break
            print('Truncating Spectral Grad Matrix to rank {} using {} threshold'.format(adaptive_rank,
                                                                                         self.adaptive_rank_th))
            U_k = U[:, 0:adaptive_rank]
            S_k = S[:adaptive_rank]
            V_k = V[0:adaptive_rank, :]
            lr_approx = np.dot(U_k * S_k, V_k)
        else:
            lr_approx = np.dot(U * S, V)

        return lr_approx

