from ftl.training_utils.trainer import Trainer
from ftl.training_utils.optimization import SchedulingOptimization
from ftl.compression.compression import Compression
from torch.utils.data import DataLoader
import numpy as np


class Client:
    def __init__(self,
                 client_id: int,
                 learner=None,
                 attack_model=None,
                 stochastic_attack=False,
                 stochastic_attack_prob=0.8,
                 C: Compression = None,
                 mal: bool = False,
                 T: float = 1.0):
        self.client_id = client_id
        self.trainer = Trainer()
        self.learner = learner

        self.mal = mal  # is it a malicious node ?
        self.attack_model = attack_model  # Ex. Type of Byzantine / poisoning attack
        self.stochastic_attack = stochastic_attack  # will this node be consistently byzantine ?
        self.attack_prob = stochastic_attack_prob

        self.C = C

        self.local_train_data = None

        self.grad = None
        self.T = T

    def client_step(self, opt_alg, opt_group, num_batches=1):
        opt = SchedulingOptimization(model=self.learner,
                                     opt_alg=opt_alg,
                                     opt_group=opt_group
                                     ).optimizer
        src_model_weights = np.concatenate([w.data.cpu().numpy().flatten() for w in self.learner.parameters()])

        # Reset gradient just in case
        self.learner.zero_grad()
        self.trainer.reset_gradient_power()
        # for bi in range(num_batches):
        local_train_loader = DataLoader(self.local_train_data.dataset, shuffle=True, batch_size=100)
        self.trainer.train(model=self.learner, optimizer=opt,
                           local_iterations=num_batches, train_loader=local_train_loader)
        # compute the local gradient
        dst_model_weights = np.concatenate([w.data.cpu().numpy().flatten() for w in self.learner.parameters()])
        self.grad = src_model_weights - dst_model_weights

    def get_stats(self):
        """
        Return (non-privacy) stats for aggregation:
          1. Sum of training losses
          2. Sum of gradients (N x mean)
          3. Sum of graident variance (N x var.)
        """
        weight = np.exp(-sum(self.trainer.epoch_losses).detach().cpu().numpy() / self.T)
        vN = self.trainer.sum_grad2 - self.trainer.sum_grad * self.trainer.sum_grad / self.trainer.counter
        return weight, self.trainer.sum_grad, vN
