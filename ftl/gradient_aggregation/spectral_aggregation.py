# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

import torch
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd


def make_positive(scale_logit):
    return torch.exp(scale_logit)


class RobustPCALayer(torch.nn.Module):
    def __init__(self, num_points, input_dim, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(RobustPCALayer, self).__init__()
        self.enc = torch.nn.Linear(input_dim, hidden_dim)
        self.dec = torch.nn.Linear(hidden_dim, input_dim)
        self.scales = torch.nn.Parameter(torch.ones(num_points))

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.dec(self.enc(x))

    def get_scales(self, indices):
        return make_positive(self.scales[indices])


class RobustPCAEstimator:
    def __init__(self, num_points, input_dim, hidden_dim, device, auto_encoder_loss):
        self.num_points = num_points
        self.pca = RobustPCALayer(num_points, input_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.pca.parameters(), lr=1e-2)
        self.auto_encoder_loss = auto_encoder_loss

    @staticmethod
    def scaled_rec_loss(x, x_hat, alphas):
        return torch.mean((x - x_hat) ** 2 / (2 * alphas.unsqueeze(1) ** 2)) + torch.mean(torch.log(alphas))

    @staticmethod
    def rec_loss(x, x_hat):
        return torch.mean((x - x_hat) ** 2)

    def _train(self, x, indices, steps=100):
        x_hat = - 999 * x
        for i in tqdm(range(steps)):
            alphas = self.pca.get_scales(indices)
            x_hat = self.pca(x)
            if self.auto_encoder_loss == 'mse':
                loss = self.rec_loss(x, x_hat)
            elif self.auto_encoder_loss == 'scaled_mse':
                loss = self.scaled_rec_loss(x, x_hat, alphas)
            else:
                raise NotImplementedError
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return ((x - x_hat) ** 2).sum(dim=1).sqrt().mean().log()

    def fit(self, x, indices, steps=4000):
        self.pca.scales.requires_grad = False
        learned_std = self._train(x, indices, steps)
        with torch.no_grad():
            # print("Learned STD: {}".format(learned_std.exp()))
            self.pca.scales.copy_(torch.ones(self.num_points) * learned_std)
        self.pca.scales.requires_grad = True
        self._train(x, indices, steps)
        return self

    def fine_tune(self, x, indices, steps=400):
        self.pca.scales.requires_grad = True
        learned_std = self._train(x, indices, steps)
        return self

    def transform(self, x, indices):
        return self.pca(x), self.pca.get_scales(indices)


def fast_lr_decomposition(X: np.ndarray,
                          rank: int = None,
                          adaptive_rank_th: float = 0.95,
                          drop_top_comp: bool = False):
    if not rank or rank > min(X.shape[0], X.shape[1]):
        rank = min(X.shape[0], X.shape[1])
    print('Doing a {} rank SVD'.format(rank))
    X = np.transpose(X)
    U, S, V = randomized_svd(X, n_components=rank,
                             flip_sign=True)
    if adaptive_rank_th:
        if not 0 < adaptive_rank_th < 1:
            raise Exception('adaptive_rank_th should be between 0 and 1')
        n_samples, n_features = X.shape
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_var = np.var(X, ddof=1, axis=0)
        explained_variance_ratio_ = explained_variance_ / total_var.sum()
        # print(explained_variance_ratio_)
        cum_var_explained = np.cumsum(explained_variance_ratio_)
        # print(cum_var_explained)
        adaptive_rank = np.searchsorted(cum_var_explained, v=adaptive_rank_th)
        if adaptive_rank == 0:
            if drop_top_comp:
                adaptive_rank += 1
            adaptive_rank += 1
        print('Truncating Spectral Grad Matrix to rank {} using '
              '{} threshold'.format(adaptive_rank, adaptive_rank_th))
        U_k = U[:, 0:adaptive_rank]
        S_k = S[0:adaptive_rank]
        V_k = V[0:adaptive_rank, :]

    else:
        U_k = U
        S_k = S
        V_k = V

    if drop_top_comp:
        U_k = U_k[:, 1:]
        S_k = S_k[1:]
        V_k = V_k[1:, :]

    lr_approx = np.dot(U_k * S_k, V_k)
    lr_approx = np.transpose(lr_approx)
    return lr_approx, S


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(999)

    num_client = 1000
    frac_adv = 0.3
    mean = 5
    std = 5

    num_corrupt = int(frac_adv * num_client)
    syn_data = torch.matmul(torch.randn(num_client, 2), torch.tensor([[.1, .8], [-.4, 1.9]])).to(device)
    ix = np.arange(len(syn_data))

    # Additive Gaussian
    if num_corrupt >= 1:
        add_noise = torch.randn(num_corrupt, 2) * std + mean
        syn_data[-num_corrupt:] = add_noise

    pca = RobustPCAEstimator(syn_data.shape[0],
                             syn_data.shape[1],
                             1,
                             device,
                             auto_encoder_loss='scaled_mse').fit(syn_data, indices=ix)

    rec, scales = pca.transform(syn_data, indices=ix)
    rec = rec.detach().cpu()
    mean_grad = torch.mean(syn_data, 0)
    print('mean_avg {}'.format(mean_grad))
    mean_spectral = torch.mean(rec, 0)
    print('mean spectral avg {}'.format(mean_spectral))

    syn_data = syn_data.detach().cpu()
    plt.scatter(syn_data[:, 0], syn_data[:, 1], c='b', s=1, label='Data Points on $\mathcal{R}^2$')
    plt.scatter(rec[:, 0], rec[:, 1], c='r', s=1, label='Projection on $\sigma_1$')
    plt.scatter(mean_grad[0], mean_grad[1], edgecolors='g', marker='*', s=200, facecolors='none',
                label='Fed Avg Estimate')
    plt.scatter(mean_spectral[0], mean_spectral[1], edgecolors='k', marker='o',
                s=50, label='Spectral Avg Estimate')
    plt.ylim(-7.5, 20)

    plt.legend(fontsize=11)
    plt.grid(axis='both')
    plt.show()
