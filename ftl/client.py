import copy
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Client:
    def __init__(self, client_id, trainer=None,
                 attack_mode=None,
                 attack_model=None,
                 stochastic_attack=False,
                 stochastic_attack_prob=0.8,
                 compression_operator='full'):
        self.client_id = client_id
        self.trainer = trainer
        self.attack_mode = attack_mode  # which attack model to use || None, byzantine, poison
        self.attack_model = attack_model  # Ex. Type of Byzantine / poisoning attack
        self.stochastic_attack = stochastic_attack  # will this node be consistently byzantine ?
        self.attack_prob = stochastic_attack_prob

        self.compression_operator = compression_operator

        self.local_train_data = None
        self.local_model = None
        self.local_model_prev = None

    def update_local_model(self, model):
        self.local_model_prev = copy.deepcopy(self.local_model)
        self.local_model = copy.deepcopy(model)

    def byzantine_update(self, w):
        # Flip a coin and decide whether to apply noise using the
        # stochastic attack probability
        # This ensures that the node while being byzantine has some
        # stochasticity i.e. it doesn't act as byzantine all the time
        if np.random.random_sample() > self.attack_prob:
            return w

        if self.attack_model == 'gaussian':
            return self._gaussian_byzantine(w=w)
        else:
            raise NotImplementedError

    def compress_message(self, w):
        if self.compression_operator == 'full':
            return w
        else:
            raise NotImplementedError

    @staticmethod
    def _gaussian_byzantine(w):
        w_attacked = copy.deepcopy(w)
        if type(w_attacked) == list:
            for k in range(len(w_attacked)):
                noise = torch.randn(w[k].shape).to(device) * w_attacked[k].to(device)
                w_attacked[k] += noise
        else:
            for k in w_attacked.keys():
                noise = torch.randn(w[k].shape).to(device) * w_attacked[k].to(device)
                w_attacked[k] += noise
        return w_attacked
