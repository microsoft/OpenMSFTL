from ftl.agents.client import Client
import numpy as np
from typing import List, Dict
import warnings


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
            warnings.warn(message='Drift Attack is implemented as a co-ordinated only attack,'
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
    Implementation of the attack mentioned (in un-coordinated setting only) in:
    Ref: Fu et.al. Attack-Resistant Federated Learning with Residual-based Reweighting
    https://arxiv.org/abs/1912.11464.

    [Our Proposal] In Co-ordinated Mode: We take the mean of all clients and generate noise based on the
    mean vector and make all the clients grad = mean(grad_i) + noise.
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'additive gaussian'
        self.noise_scale = attack_config["noise_scale"]

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) == 0:
            return
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
    """
    Random Gaussian Noise as used in the paper (in the uncoordinated setting)
    Ref: Cong et.al. Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance (ICML'19).

    [Our Proposal] In the co-ordinate setting all the mal client's grad vectors are set to the same,
    drawn randomly from a Normal Distribution with zero mean and specified std
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'random gaussian'
        self.attack_std = attack_config["attack_std"]

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) == 0:
            return

        byz_grad = np.random.normal(loc=0, scale=self.attack_std,
                                    size=byz_clients[0].grad.shape).astype(byz_clients[0].grad.dtype)
        for client in byz_clients:
            client.grad = byz_grad
