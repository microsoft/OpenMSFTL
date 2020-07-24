from ftl.training_utils.trainer import Trainer
from ftl.training_utils.optimization import SchedulingOptimization
from ftl.comm_compression.compression import Compression
import numpy as np


class Client:
    def __init__(self,
                 client_id,
                 learner=None,
                 attack_model=None,
                 stochastic_attack=False,
                 stochastic_attack_prob=0.8,
                 C: Compression = None):

        self.client_id = client_id
        self.trainer = Trainer()
        self.learner = learner

        self.mal = False  # is it a malicious node ?
        self.attack_model = attack_model  # Ex. Type of Byzantine / poisoning attack
        self.stochastic_attack = stochastic_attack  # will this node be consistently byzantine ?
        self.attack_prob = stochastic_attack_prob

        self.C = C

        self.local_train_data = None

        self.grad = None

    def client_step(self, opt_alg, opt_group, num_batches=1):
        opt = SchedulingOptimization(model=self.learner,
                                     opt_alg=opt_alg,
                                     opt_group=opt_group
                                     ).optimizer

        src_model_weights = np.concatenate([w.data.cpu().numpy().flatten() for w in self.learner.parameters()])
        # Reset gradient just in case
        self.learner.zero_grad()
        for bi in range(num_batches):
            self.trainer.train(model=self.learner,
                               optimizer=opt)
        # compute the local gradient
        dst_model_weights = np.concatenate([w.data.cpu().numpy().flatten() for w in self.learner.parameters()])
        self.grad = src_model_weights - dst_model_weights



