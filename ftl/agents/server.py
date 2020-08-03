import random
from typing import Dict
from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from ftl.attacks import launch_attack
from ftl.gradient_aggregation.aggregation import Aggregator
from ftl.models.model_helper import dist_weights_to_model, dist_grads_to_model
from ftl.training_utils import infer
from ftl.training_utils.optimization import SchedulingOptimization
from .client import Client

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Server:
    def __init__(self,
                 server_model,
                 aggregator_config: Dict,
                 server_opt_config: Dict = None,
                 server_lrs_config: Dict = None,
                 clients: List[Client] = None,
                 val_loader: DataLoader = None,
                 test_loader: DataLoader = None):

        self.learner = server_model
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.learner.to('cpu').parameters()])
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.clients = clients
        self.aggregator = Aggregator(aggregation_config=aggregator_config)
        opt_obj = SchedulingOptimization(model=self.learner,
                                         opt_group=server_opt_config,
                                         lrs_group=server_lrs_config)
        self.opt = opt_obj.optimizer
        self.lrs = opt_obj.lr_scheduler
        self.test_acc = []
        self.val_acc = []
        self.train_loss = []
        self.agg_grad = None

    def init_client_models(self):
        for client in self.clients:
            dist_weights_to_model(weights=self.w_current, parameters=client.learner.parameters())

    def train_client_models(self, num_participating_client: int,
                            client_config: Dict = None,
                            attack_config: Dict = None):
        """
        Update each client model
        :param attack_config:
        :param num_participating_client: number of clients to be selected
        :param client_config: specifying parameters for client trainer
        """
        # Sample Clients to Train this round
        sampled_clients = random.sample(population=self.clients, k=num_participating_client)
        epoch_loss = 0.0
        mal_nodes = []
        for ix, client in enumerate(sampled_clients):
            client.client_step(num_batches=client_config['num_batches'])
            epoch_loss += client.trainer.epoch_losses[-1]
            if client.mal:
                mal_nodes.append(client)
        train_loss = epoch_loss / len(sampled_clients)
        self.train_loss.append(train_loss.item())

        # Modify the gradients of malicious nodes if attack is defined
        if len(mal_nodes) > 0:
            launch_attack(attack_mode=attack_config["attack_mode"], mal_nodes=mal_nodes)

        self.agg_grad = self.aggregator.get_aggregate(clients=sampled_clients, alphas=None)

    def update_global_model(self):
        print('server lr = {}'.format(self.lrs.get_lr()))
        dist_grads_to_model(grads=self.agg_grad, parameters=self.learner.to('cpu').parameters())
        self.opt.step()
        self.lrs.step()
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.learner.to('cpu').parameters()])
        dist_weights_to_model(weights=self.w_current, parameters=self.learner.to('cpu').parameters())

    def run_validation(self):
        """
        Run validation with the current model
        """
        val_acc, _ = infer(test_loader=self.val_loader, model=self.learner)
        return val_acc

    def run_test(self):
        """
        Run test with the current model
        """
        test_acc, _ = infer(test_loader=self.test_loader, model=self.learner)
        return test_acc
