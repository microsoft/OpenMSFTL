from ftl.trainer import Trainer
from ftl.optimization import Optimization, _get_lr
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

        self.grads = None

    def train_step(self, lr, reg, iterations):
        opt = Optimization(model=self.learner,
                           lr0=lr,
                           reg=reg).optimizer
        self.trainer.train(data=self.local_train_data,
                           model=self.learner,
                           optimizer=opt,
                           epochs=iterations)
        # Now, we must have done iterations number of gradient steps
        self.grads = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.learner.parameters()])

    # def byzantine_update(self, w):
    #     # Flip a coin and decide whether to apply noise using the
    #     # stochastic attack probability
    #     # This ensures that the node while being byzantine has some
    #     # stochasticity i.e. it doesn't act as byzantine all the time
    #     if np.random.random_sample() > self.attack_prob:
    #         return w
    #
    #     if self.attack_model == 'gaussian':
    #         return self._gaussian_byzantine(w=w)
    #     else:
    #         raise NotImplementedError
    #
    # @staticmethod
    # def _gaussian_byzantine(w):
    #     w_attacked = copy.deepcopy(w)
    #     if type(w_attacked) == list:
    #         for k in range(len(w_attacked)):
    #             noise = torch.randn(w[k].shape).to(device) * w_attacked[k].to(device)
    #             w_attacked[k] += noise
    #     else:
    #         for k in w_attacked.keys():
    #             noise = torch.randn(w[k].shape).to(device) * w_attacked[k].to(device)
    #             w_attacked[k] += noise
    #     return w_attacked
