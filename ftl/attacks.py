import numpy as np


class ByzAttack:
    """
    This class implements several algorithms to modify the gradients of
    byzantine nodes using different strategies.

    In particular this has implementations of the following algorithms:

    1. Gilad Baruch et.al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
       Ref: https://github.com/moranant/attacking_distributed_learning

    """
    def __init__(self, k: int = 2):
        self.grad_mean = None
        self.grad_std = None
        self.k = k  # k defines how many std away from mean

    def attack(self, byz_clients):
        if len(byz_clients) == 0 or self.k == 0:
            return

        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)

        self.grad_mean = np.mean(clients_grad, axis=0)
        self.grad_std = np.var(clients_grad, axis=0) ** 0.5

        # compute corruption [ \mu - k * \sigma ]
        byz_grad = self.grad_mean[:] - self.k * self.grad_std[:]

        # update the malicious client grads
        for client in byz_clients:
            client.grad = byz_grad

