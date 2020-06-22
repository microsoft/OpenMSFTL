import copy
import torch
from typing import List


class Client:
    def __init__(self, client_id, trainer=None, adv_noise=None):
        self.client_id = client_id
        self.trainer = trainer
        self.adversary_mode = adv_noise
        self.local_train_data = None
        self.local_model = None
        self.local_model_prev = None

    def update_local_model(self, model):
        self.local_model_prev = copy.deepcopy(self.local_model)
        self.local_model = copy.deepcopy(model)


class Server:
    def __init__(self, aggregation='fed_avg'):
        self.val_loader = None
        self.test_loader = None
        self.aggregation = None
        self.global_model = None
        self.test_acc = 0.0
        self.val_acc = []
        self.train_loss = []
        self.aggregation = aggregation

    def aggregate(self, clients):
        if self.aggregation == 'fed_avg':
            agg_params = self._fed_average(clients=clients)
        else:
            raise NotImplementedError

        self.global_model.load_state_dict(agg_params)

    @staticmethod
    def _fed_average(clients: List[Client]):
        epoch_params = []
        for client in clients:
            epoch_params.append(copy.deepcopy(client.local_model.state_dict()))

        server_param = copy.deepcopy(epoch_params[0])
        for key in server_param.keys():
            for i in range(1, len(epoch_params)):
                server_param[key] += epoch_params[i][key]
            server_param[key] = torch.div(server_param[key], len(epoch_params))

        return server_param





