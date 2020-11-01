# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

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
        self.attack_std = attack_config["attack_std"]
        self.noise_scale = attack_config["noise_scale"]
        self.fixed = attack_config["fixed"]
        self.to_scale = attack_config["to_scale"]

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) == 0:
            return
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)
        # apply gaussian noise (scaled appropriately)
        if self.fixed:
            np.random.seed(999)
        noise = np.random.normal(loc=self.noise_scale*grad_mean if self.to_scale else self.noise_scale,
                                 scale=self.attack_std,
                                 size=grad_mean.shape).astype(dtype=grad_mean.dtype)
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
        self.noise_scale = attack_config["noise_scale"]
        self.fixed = attack_config["fixed"]
        self.to_scale = attack_config["to_scale"]

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) == 0:
            return
        if self.noise_scale == 0.0:
            noise_loc = np.zeros(len(byz_clients[0].grad), dtype=byz_clients[0].grad.dtype)
        elif self.noise_scale != 0.0:
            clients_grad = [client.grad for client in byz_clients]
            noise_loc = np.mean(clients_grad, axis=0)

        # apply gaussian noise (scaled appropriately)
        if self.fixed:
            np.random.seed(999)
        noise = np.random.normal(loc=self.noise_scale*noise_loc if self.to_scale else self.noise_scale,
                                 scale=self.attack_std,
                                 size=noise_loc.shape).astype(dtype=noise_loc.dtype)
        for client in byz_clients:
            client.grad = noise


class BitFlipAttack(ByzAttack):
    """
    The bits that control the sign of the floating numbers are flipped, e.g.,
    due to some hardware failure. A faulty worker pushes the negative gradient instead
    of the true gradient to the servers.
    In Co-ordinated mode: one of the faulty gradients is copied to and overwrites the other faulty gradients,
    which means that all the faulty gradients have the same value = - mean(g_i) ; i in byz clients

    Ref: Cong et.al. Zeno: Distributed Stochastic Gradient Descent with Suspicion-based Fault-tolerance (ICML'19).
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'bit flip attack'

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) == 0:
            return
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)
        for client in byz_clients:
            client.grad = - grad_mean


class RandomSignFlipAttack(ByzAttack):
    """
    A faulty worker randomly (binomial) flips the sign of its each gradient co-ordinates
    In Co-ordinate Mode: We do the same to the mean of all the byz workers and all workers are
    assigned the same faulty gradient.
    Ref: Bernstein et.al. SIGNSGD WITH MAJORITY VOTE IS COMMUNICATION EFFICIENT AND FAULT TOLERANT ; (ICLR '19)
    """

    def __init__(self, attack_config: Dict):
        ByzAttack.__init__(self, attack_config=attack_config)
        self.attack_algorithm = 'sign flip attack'

    def attack(self, byz_clients: List[Client]):
        if len(byz_clients) == 0:
            return
        clients_grad = []
        for client in byz_clients:
            clients_grad.append(client.grad)
        grad_mean = np.mean(clients_grad, axis=0)
        faulty_grad = np.zeros_like(grad_mean)
        for i in range(0, len(grad_mean)):
            faulty_grad[i] = grad_mean[i] if np.random.random() < 0.5 else -grad_mean[i]
        for client in byz_clients:
            client.grad = faulty_grad
