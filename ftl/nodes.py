class Client:
    def __init__(self, client_id, adv_noise=None):
        self.client_id = client_id
        self.local_train_data = None
        self.adversary_mode = adv_noise
        self.local_model_t = None
        self.local_model_prev = None

    def _update_local_model(self, model):
        self.local_model_prev = self.local_model
        self.local_model = model


class Server:
    def __init__(self):
        self.val_loader = None
        self.test_loader = None
        self.aggregation = None
        self.global_model = None

