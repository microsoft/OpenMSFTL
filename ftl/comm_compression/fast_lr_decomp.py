from sklearn.utils.extmath import randomized_svd
import numpy as np


class FastLRDecomposition:
    def __init__(self, n_components, X: np.ndarray = None, copy=True, whiten=False, tol=0.0, iterated_power='auto'):
        self.n_components = n_components
        self.X = X
        self.copy = copy
        self.whiten = whiten

        self.tol = tol
        self.iterated_power = iterated_power

        self.Sigma = []

        self.explained_variance_ = []
        self.explained_variance_ratio_ = []
        self.noise_variance_ = []
        self.agg_grad = None

        if X is not None:
            self.decompose(X=X)

    def decompose(self, X):
        n_samples, n_features = X.shape
        # sign flipping is done inside
        U, S, V = randomized_svd(X, n_components=self.n_components,
                                 n_iter=self.iterated_power,
                                 flip_sign=True)

        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.var(X, ddof=1, axis=0)
        explained_variance_ratio_ = explained_variance_ / total_var.sum()

        if self.n_components < min(n_features, n_samples):
            noise_variance_ = (total_var.sum() - explained_variance_.sum())
            noise_variance_ /= min(n_features, n_samples) - self.n_components
            self.noise_variance_.append(noise_variance_)
        else:
            self.noise_variance_.append('nan')
            pass

        self.explained_variance_.append(explained_variance_)
        self.explained_variance_ratio_.append(explained_variance_ratio_)

        self.Sigma.append(S)
        self.agg_grad = np.mean(np.dot(U * S, V), axis=0)




