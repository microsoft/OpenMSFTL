import copy
import torch


class Client:
    def __init__(self, client_id, trainer=None, adv_noise=None, stochastic_attack=False):
        self.client_id = client_id
        self.trainer = trainer
        self.adversary_mode = adv_noise  # which attack model to use
        self.attack_strategy = stochastic_attack  # will this node be consistently byzantine ?

        self.local_train_data = None
        self.local_model = None
        self.local_model_prev = None

    def update_local_model(self, model):
        self.local_model_prev = copy.deepcopy(self.local_model)
        self.local_model = copy.deepcopy(model)

        if self.adversary_mode:
            raise NotImplementedError

    def byzantine_update(self, w):
        raise NotImplementedError

    @staticmethod
    def _gaussian_byzantine(w, scale):
        w_attacked = copy.deepcopy(w)
        if type(w_attacked) == list:
            for k in range(len(w_attacked)):
                noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
                w_attacked[k] += noise
        else:
            for k in w_attacked.keys():
                noise = torch.randn(w[k].shape).cuda() * scale / 100.0 * w_attacked[k]
                w_attacked[k] += noise
        return w_attacked
