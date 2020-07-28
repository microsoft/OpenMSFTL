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
        self.G = None
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
        self.Sigma = S
        # Normalize Sigma
        self.normalized_Sigma = S / sum(S)

        if self.adaptive_k_th:
            running_pow = 0.0
            adaptive_k = 0
            for sv in self.normalized_Sigma:
                running_pow += sv
                adaptive_k += 1
                if running_pow >= self.adaptive_k_th:
                    break
            print('Truncating Spectral Grad Matrix to rank {} using {} threshold'.format(adaptive_k,
                                                                                         self.adaptive_k_th))
            U_k = U[:, 0:adaptive_k]
            S_k = S[:adaptive_k]
            V_k = V[0:adaptive_k, :]
            lr_approx = np.dot(U_k * S_k, V_k)
        else:
            lr_approx = np.dot(U * S, V)

        self.G = lr_approx







