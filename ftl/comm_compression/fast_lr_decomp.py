from sklearn.utils.extmath import randomized_svd
import numpy as np


class FastLRDecomposition:
    def __init__(self, n_components,
                 X: np.ndarray = None,
                 iterated_power='auto',
                 adaptive_k_th=None):
        self.n_components = n_components
        self.X = X
        self.iterated_power = iterated_power
        self.Sigma = []
        self.normalized_Sigma = None
        # self.explained_variance_ = None
        # self.explained_variance_ratio_ = None
        # self.noise_variance_ = None
        self.agg_grad = None
        self.adaptive_k_th = adaptive_k_th

        if X is not None:
            self.decompose(X=X)

    def decompose(self, X):
        # n_samples, n_features = X.shape
        # sign flipping is done inside
        print('Doing a {} rank SVD'.format(self.n_components))
        U, S, V = randomized_svd(X, n_components=self.n_components,
                                 n_iter=self.iterated_power,
                                 flip_sign=True)

        # explained_variance_ = (S ** 2) / (n_samples - 1)
        # total_var = np.var(X, ddof=1, axis=0)
        # explained_variance_ratio_ = explained_variance_ / total_var.sum()
        #
        # if self.n_components < min(n_features, n_samples):
        #     noise_variance_ = (total_var.sum() - explained_variance_.sum())
        #     noise_variance_ /= min(n_features, n_samples) - self.n_components
        #     self.noise_variance_ = noise_variance_
        # else:
        #     pass
        #
        # self.explained_variance_ = explained_variance_
        # self.explained_variance_ratio_ = explained_variance_ratio_

        self.Sigma = S

        if self.adaptive_k_th:
            # Normalize Sigma and choose rank to keep adaptively
            self.normalized_Sigma = S / sum(S)
            running_pow = 0.0
            adaptive_k = 0

            for sv in self.normalized_Sigma:
                running_pow += sv
                adaptive_k += 1
                if running_pow >= self.adaptive_k_th:
                    break

            print('Truncating Spectral Grad Matrix to rank {} using {} threshold'.format(adaptive_k, self.adaptive_k_th))
            U_k = U[:, 0:adaptive_k]
            S_k = S[:adaptive_k]
            V_k = V[0:adaptive_k, :]
            self.agg_grad = np.mean(np.dot(U_k * S_k, V_k), axis=0)
        else:
            self.agg_grad = np.mean(np.dot(U * S, V), axis=0)




