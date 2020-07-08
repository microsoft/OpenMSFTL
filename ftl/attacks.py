from ftl.client import Client
import numpy as np
from typing import List


class Attack:
    """
    This class implements several algorithms to modify the gradients of
    byzantine nodes using different strategies.
    """
    def __init__(self, k: float = 1.5, attack_model: str = 'drift'):
        """
        :param k: specify how many standard dev away byz nodes grad mean is from true grad mean
        :param attack_model: specify the type of attack Options: 'drift'
        """
        self.k = k
        self.attack_model = attack_model

    def attack(self, byz_clients: List[Client]):
        print("{} Attack Enabled in {} clients ".format(self.attack_model, len(byz_clients)))
        if len(byz_clients) == 0 or self.k == 0:
            return
        if self.attack_model is 'drift':
            byz_grad = self.drift_attack(k=self.k, byz_clients=byz_clients)
        else:
            raise NotImplementedError
        # update the malicious client grads
        for client in byz_clients:
            client.grad = byz_grad

    @staticmethod
    def drift_attack(k: float, byz_clients: List[Client]) -> np.ndarray:
        """
        Implementation of the powerful drift attack algorithm proposed in:
        Gilad Baruch et.al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
        Ref: https://github.com/moranant/attacking_distributed_learning
        """
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)
        grad_std = np.var(clients_grad, axis=0) ** 0.5
        # compute corruption [ \mu - k * \sigma ]
        return grad_mean[:] - k * grad_std[:]


