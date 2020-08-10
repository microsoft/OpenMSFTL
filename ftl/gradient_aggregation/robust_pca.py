# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from tqdm import tqdm


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

    def get_scales(self, indicies):
        return make_positive(self.scales[indicies])


class RobustPCAEstimator:

    def __init__(self, num_points, input_dim, hidden_dim, device):
        self.num_points = num_points
        self.pca = RobustPCALayer(num_points, input_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.pca.parameters(), lr=1e-2)

    def scaled_rec_loss(self, x, x_hat, scales):
        return torch.mean((x - x_hat) ** 2 / (2 * scales.unsqueeze(1) ** 2)) \
               + torch.mean(torch.log(scales))

    def _train(self, x, indicies, steps):
        for i in tqdm(range(steps)):
            scales = self.pca.get_scales(indicies)
            x_hat = self.pca(x)
            loss = self.scaled_rec_loss(x, x_hat, scales)
            #if i % 50 == 0:
            #    print(scales.min().item(), scales.mean().item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return ((x - x_hat) ** 2).sum(dim=1).sqrt().mean().log()

    def fit(self, x, indicies, steps=2000):
        self.pca.scales.requires_grad = False
        learned_std = self._train(x, indicies, steps)
        with torch.no_grad():
            print("Learned STD: {}".format(learned_std.exp()))
            self.pca.scales.copy_(torch.ones(self.num_points) * learned_std)
        self.pca.scales.requires_grad = True
        self._train(x, indicies, steps)
        return self

    def fine_tune(self, x, indicies, steps=400):
        self.pca.scales.requires_grad = True
        self._train(x, indicies, steps)
        return self

    def transform(self, x, indicies):
        return self.pca(x), self.pca.get_scales(indicies)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = "cuda:0"

    x = torch.matmul(torch.randn(100, 2), torch.tensor([[.1, .8], [-.4, 1.9]])).to(device)
    x[-4:] = 10

    pca = RobustPCAEstimator(x.shape[0], x.shape[1], 1, device).fit(x)

    rec, scales = pca.transform(x)
    rec = rec.detach().cpu()
    x = x.detach().cpu()
    plt.plot(x[:, 0], x[:, 1], "k.")
    plt.plot(rec[:, 0], rec[:, 1], "r.")
    plt.show()
