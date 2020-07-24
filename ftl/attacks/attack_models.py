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
    Ref: Gilad Baruch et.al. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
    https://github.com/moranant/attacking_distributed_learning
    """
    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'drift'
        self.n_std = attack_config["attack_n_std"]

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) <= 1 or self.n_std == 0:
            print('Drift Attack is implemented as a co-ordinated only attack,'
                  'In the un-coordinated mode we are leaving the grads unchanged')
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
    """
    Additive Gaussian Noise, scaled w.r.t the original values.
    Implementation of the attack mentioned in:
    Ref: Fu et.al. Attack-Resistant Federated Learning with Residual-based Reweighting
    https://arxiv.org/abs/1912.11464

    [Our Proposal] In Co-ordinated Mode: We take the mean of all clients and generate noise based on the
    mean vector and make all the clients grad = mean(grad_i) + noise.
    In the un-coordinated mode
    """
    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'additive gaussian'
        self.noise_scale = attack_config["noise_scale"]

    def attack(self, byz_clients: List[Client]):
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)

        # apply gaussian noise (scaled appropriately)
        noise = np.random.normal(loc=0.0, scale=1.0, size=grad_mean.shape).astype(dtype=grad_mean.dtype)
        noise *= self.noise_scale * grad_mean
        byz_grad = grad_mean[:] + noise

        for client in byz_clients:
            client.grad = byz_grad


class RandomGaussian(ByzAttack):
    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'random gaussian'
