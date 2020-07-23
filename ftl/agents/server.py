from .client import Client
from ftl.training_utils.optimization import get_lr
from ftl.models.model_helper import dist_weights_to_model
from ftl.gradient_aggregation.aggregation import Aggregator
from typing import List
from typing import Dict
from torch.utils.data import DataLoader
import random


class Server:
    def __init__(self,
                 model,
                 aggregation_scheme: str = 'fed_avg',
                 rank: int = 10,
                 krum_frac: float = 0.7,
                 optimizer_scheme: str = None,
                 server_config: Dict = None,
                 clients: List[Client] = None,
                 val_loader: DataLoader = None,
                 test_loader: DataLoader = None):

        if not server_config:
            server_config = {"lr0": 1.0, "lr_restart": 100, "lr_decay": 10}

        # keep server parameters
        self.server_lr0 = server_config["lr0"]
        self.server_lr_restart = server_config["lr_restart"]
        self.lrs = server_config["lr_schedule"]
        self.lr_decay = server_config["lr_decay"]
        self.num_rounds = 0

        # Server has access to Test and Dev Data Sets to evaluate Training Process
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Server has a pointer to all clients
        self.clients = clients
        self.current_lr = self.server_lr0

        # Aggregator tracks the model and optimizer
        self.aggregator = Aggregator(agg_strategy=aggregation_scheme,
                                     rank=rank,
                                     m_krum=krum_frac,
                                     model=model,
                                     dual_opt_alg=optimizer_scheme,
                                     opt_group={'lr': self.current_lr, 'lrs': self.lrs})

        # Server only keeps track of the pointer to the updated weights at each round
        self.w_current = self.aggregator.w_current

        # Containers to store metrics
        self.test_acc = []
        self.val_acc = []
        self.train_loss = []

    def get_global_model(self):
        """
        return: global model
        """
        return self.aggregator.model

    def init_client_models(self):
        # Loop over all clients and update the current params
        for client in self.clients:
            # client.w_init = copy.deepcopy(self.w_current)
            dist_weights_to_model(weights=self.w_current, parameters=client.learner.parameters())

    def _update_server_lr(self):
        if self.num_rounds % self.server_lr_restart == 0:
            self.current_lr = self.server_lr0 / self.lr_decay
            self.aggregator.set_lr(self.current_lr)

        # take a step in lr_scheduler
        if self.aggregator.lr_scheduler is not None:
            self.aggregator.lr_scheduler.step()
            self.current_lr = get_lr(self.aggregator.optimizer)

        print('current lr = {}'.format(self.current_lr))

    def train_client_models(self, k: int, client_config: Dict = None, attacker=None):
        """
        Update each client model
        :param attacker: Attack Object
        :param k: number of clients to be selected
        :param client_config: specifying parameters for client trainer
        """
        if not client_config:
            client_config = {'optimizer_scheme': 'SGD',
                             'lr': 0.002,
                             'weight_decay': 0.0,
                             'momentum': 0.9,
                             'num_batches': 1}
        # Sample Clients to Train this round
        sampled_clients = random.sample(population=self.clients, k=k)

        # Now we will loop through these clients and do training steps
        # Compute number of local gradient steps per communication round
        epoch_loss = 0.0

        # TODO : Parallel Calls (Defer)
        for client in sampled_clients:
            client.client_step(opt_alg=client_config['optimizer_scheme'],
                               opt_group={'lr': client_config['lr'],
                                          'weight_decay': client_config['weight_decay'],
                                          'momentum': client_config['momentum']},
                               num_batches=client_config['num_batches'])
            # print('Client : {} loss = {}'.format(client.client_id, client.trainer.epoch_losses[-1]))
            epoch_loss += client.trainer.epoch_losses[-1]

        # Modify the gradients of malicious nodes if attack is defined
        if attacker:
            # Co-ordinated Attack
            mal_nodes = [c for c in sampled_clients if c.attack_mode]
            if mal_nodes:
                attacker.attack(byz_clients=mal_nodes)

        # now we can apply the compression operator before communicating to Server
        for client in sampled_clients:
            client.grad = client.C.compress(grad=client.grad)

        # Update Metrics
        train_loss = epoch_loss / len(sampled_clients)
        self.train_loss.append(train_loss.item())

        # update the learning rate: self.current_lr
        self.num_rounds += 1
        self._update_server_lr()

        # aggregate client updates
        self.w_current = self.aggregator.update_model(clients=sampled_clients, current_lr=self.current_lr)
