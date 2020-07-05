from ftl.trainer import Trainer
from ftl.optimization import SchedulingOptimizater
from ftl.compression import Compression
import numpy as np


class Client:
    def __init__(self,
                 client_id,
                 learner=None,
                 attack_mode=None,
                 attack_model=None,
                 stochastic_attack=False,
                 stochastic_attack_prob=0.8,
                 C: Compression = None):

        self.client_id = client_id
        self.trainer = Trainer()
        self.learner = learner

        self.attack_mode = attack_mode  # which attack model to use || None, byzantine, poison
        self.attack_model = attack_model  # Ex. Type of Byzantine / poisoning attack
        self.stochastic_attack = stochastic_attack  # will this node be consistently byzantine ?
        self.attack_prob = stochastic_attack_prob

        self.C = C

        self.local_train_data = None

        self.grad = None

    def client_step(self, opt_alg, opt_group):
        opt = SchedulingOptimizater(model=self.learner,
                                    opt_alg=opt_alg,
                                    opt_group=opt_group
                                    ).optimizer

        self.trainer.train(model=self.learner,
                           optimizer=opt)
        # TODO: change this naive way since keeping the graph consumes a lot of meomry
        # Accumulate the gradient learnt
        self.grad = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.learner.parameters()])
        # now we can apply the compression operator before passing to Server
#        self.grad = self.C.compress(w=self.grad)




