from ftl.agg_utils import weighted_average
import numpy as np
import warnings


class Attack:
    """
    This class implements several algorithms to modify the gradients of
    byzantine nodes using different strategies.



    """
    def __init__(self, k: float = 1.5, attack_model: str = 'drift'):
        self.grad_mean = None
        self.grad_std = None
        self.k = k  # k defines how many std away from mean
        self.attack_model = attack_model

    def attack(self, byz_clients):
        if len(byz_clients) == 0 or self.k == 0:
            warnings.warn(" Applying attack failed. Please check Attack.attack ")
            return

        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)

        self.grad_mean = np.mean(clients_grad, axis=0)
        self.grad_std = np.var(clients_grad, axis=0) ** 0.5

        if self.attack_model is 'drift':
            # compute corruption [ \mu - k * \sigma ]
            byz_grad = self.drift(k=self.k, grad_mean=self.grad_mean, grad_std=self.grad_std)
        else:
            raise NotImplementedError

        # update the malicious client grads
        for client in byz_clients:
            client.grad = byz_grad

    @staticmethod
    def drift(k, grad_mean, grad_std):
        """
        Implementation of the powerful drift attack algorithm proposed in:
        Gilad Baruch et.al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
        Ref: https://github.com/moranant/attacking_distributed_learning
        """
        # Drift Attack | " A Little Is Enough: Circumventing Defenses For Distributed Learning NeuRips 2019 "
        return grad_mean[:] - k * grad_std[:]


