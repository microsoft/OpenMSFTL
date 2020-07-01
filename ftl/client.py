from ftl.trainer import Trainer
from ftl.optimization import Optimization
import numpy as np


class Client:
    def __init__(self,
                 client_id,
                 learner=None,
                 attack_mode=None,
                 attack_model=None,
                 stochastic_attack=False,
                 stochastic_attack_prob=0.8,
                 compression_operator='full'):

        self.client_id = client_id
        self.trainer = Trainer()
        self.learner = learner

        self.attack_mode = attack_mode  # which attack model to use || None, byzantine, poison
        self.attack_model = attack_model  # Ex. Type of Byzantine / poisoning attack
        self.stochastic_attack = stochastic_attack  # will this node be consistently byzantine ?
        self.attack_prob = stochastic_attack_prob

        self.compression_operator = compression_operator

        self.local_train_data = None

        self.grad = None

    def client_step(self, lr, reg, momentum):
        opt = Optimization(model=self.learner,
                           lr=lr,
                           reg=reg,
                           momentum=momentum).optimizer

        self.trainer.train(model=self.learner,
                           optimizer=opt)
        # Now, we must have done iterations number of gradient steps
        self.grad = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.learner.parameters()])




