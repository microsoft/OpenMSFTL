from ftl.agents.client import Client
from .attack_definition import ByzAttackCoordinated
import numpy as np
from typing import List


def get_attack(args):
    if args.attack_model == 'drift':
        return DriftAttack(std=args.attack_std)
    else:
        return None


class DriftAttack(ByzAttackCoordinated):
    """
    Implementation of the powerful drift attack algorithm proposed in:
    Gilad Baruch et.al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
    Ref: https://github.com/moranant/attacking_distributed_learning
    """
    def __init__(self, std):
        ByzAttackCoordinated.__init__(self, std=std)
        self.attack_algorithm = 'drift'

    def attack(self, byz_clients: List[Client]):

        if len(byz_clients) == 0 or self.std == 0:
            return
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)
        grad_std = np.var(clients_grad, axis=0) ** 0.5

        # apply grad corruption = [ \mu - std * \sigma ]
        byz_grad = grad_mean[:] - self.std * grad_std[:]
        for client in byz_clients:
            client.grad = byz_grad
