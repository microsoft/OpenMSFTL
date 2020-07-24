from ftl.agents.client import Client
import numpy as np
from typing import List, Dict


class ByzAttack:
    """ This is the Base Class for Byzantine attack. """
    def __init__(self, attack_config: Dict):
        self.attack_config = attack_config

    def attack(self, byz_clients: List[Client]):
        pass


class DriftAttack(ByzAttack):
    """
    Implementation of the powerful drift attack algorithm proposed in:
    Gilad Baruch et.al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
    Ref: https://github.com/moranant/attacking_distributed_learning
    """
    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'drift'
        self.n_std = attack_config["attack_n_std"]

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) == 0 or self.n_std == 0:
            return
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)
        grad_std = np.var(clients_grad, axis=0) ** 0.5

        # apply grad corruption = [ \mu - std * \sigma ]
        byz_grad = grad_mean[:] - self.n_std * grad_std[:]
        for client in byz_clients:
            client.grad = byz_grad


class AdditiveGaussian(ByzAttack):
    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'additive gaussian'

    def attack(self, byz_clients: List[Client]):
        pass


class RandomGaussian(ByzAttack):
    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'random gaussian'
