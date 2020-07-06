import numpy as np


class ByzAttack:
    """
    This class implements several algorithms to modify the gradients of
    byzantine nodes using different strategies.

    In particular this has implementations of the following algorithms:

    1. Gilad Baruch et.al. A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
       https://github.com/moranant/attacking_distributed_learning

    """
    def __init__(self):
        self.grad_mean = None
        self.grad_std = None

    def attack(self, clients):
        if len(clients) == 0:
            return

        clients_grad = []
        for client in clients:
            clients_grad.append(client.grad)

        self.grad_mean = np.mean(clients_grad, axis=0)
        self.grad_std = np.var(clients_grad, axis=0) ** 0.5


