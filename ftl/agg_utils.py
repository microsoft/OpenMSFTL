from collections import defaultdict
import numpy as np

# ------------------------------------------------- #
#             GAR implementation utils              #
# ------------------------------------------------- #


def weighted_average(clients, alphas=None):
    """
    Implements weighted average of client grads if no weights are supplied
    then its equivalent to simple average / Fed Avg
    """
    if alphas is None:
        alphas = [1] * len(clients)
    agg_grad = np.zeros_like(clients[0].grad)
    tot = np.sum(alphas)
    for alpha, client in zip(alphas, clients):
        agg_grad += (alpha / tot) * client.grad

    return agg_grad


def geo_median(clients):
    """
    Computes Geometric median of given points (vectors) using a Alternating Minimization
    based numerically stable variant of Weiszfeld's Algorithm.
    """
    pass


def get_krum_dist(clients) -> defaultdict:
    """ Computes a dist matrix between each pair of client based on grad value """
    dist = defaultdict(dict)
    for i in range(len(clients)):
        for j in range(i):
            dist[i][j] = dist[j][i] = np.linalg.norm(clients[i].grad - clients[j].grad)
    return dist
