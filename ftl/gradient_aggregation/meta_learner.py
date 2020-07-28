import numpy as np
from typing import List
from torch import nn


class GradLearner:
    """
    Given n client gradient estimates g_i for i in (1, n)
    usually fed avg would update its weight estimate as w_t+1 = w_t - lr * F(g_i)
    F operator is known as the Gradient Aggregation Rule (GAR)
    For vanilla Fed Avg F is the mean operator.

    Here, we implement a variant where we learn a more general combination using validation data.
    """

    def __init__(self, model_arch: str = 'linear'):
        self.model_arch = model_arch


class LinearEstimator(nn.Module):
    """
    Simple Linear Combination of gradient vectors to minimize validation loss
    min [ Loss ( y_val, < x_val , w_t - eta * sum (alpha_i * g_i) > ) ]
    """

    def __init__(self, dim_in: int, dim_out: int):
        super(LinearEstimator, self).__init__()
        # dim in is number of clients
        # dim out is number of labels in data_set
        self.fc_in = nn.Linear(dim_in, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, w_t, eta: float):
        # x should have shape number of grad_dim * num_clients
        x = self.fc_in(x)
        z = self.softmax(x)

        return z
