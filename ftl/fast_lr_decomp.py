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

        self.U = None
        self.Sigma = None
        self.V = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = 0
        self.agg_grad = None

        if X:
            self.decompose(X=X)

    def decompose(self, X):
        n_samples, n_features = X.shape
        # sign flipping is done inside
        U, S, V = randomized_svd(X, n_components=self.n_components,
                                 n_iter=self.iterated_power,
                                 flip_sign=True)
        self.explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.var(X, ddof=1, axis=0)
        self.explained_variance_ratio_ = \
            self.explained_variance_ / total_var.sum()

        self.U = U
        self.Sigma = S
        self.V = V

        if self.n_components < min(n_features, n_samples):
            self.noise_variance_ = (total_var.sum() -
                                    self.explained_variance_.sum())
            self.noise_variance_ /= min(n_features, n_samples) - self.n_components
        else:
            pass




