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
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0
        self.curr_client_losses = None
        self.agg_grad = None

    def init_client_models(self):
        for client in self.clients:
            dist_weights_to_model(weights=self.w_current, parameters=client.learner.parameters())

    def train_client_models(self, num_participating_client: int,
                            attack_config: Dict = None):
        """
        Update each client model
        :param attack_config:
        :param num_participating_client: number of clients to be selected
        """
        # Sample Clients to Train this round
        sampled_clients = random.sample(population=self.clients, k=num_participating_client)
        self.curr_client_losses = []
        mal_nodes = []
        for ix, client in enumerate(sampled_clients):
            client.client_step()
            self.curr_client_losses.append(client.trainer.epoch_losses[-1].item())
            if client.mal:
                mal_nodes.append(client)
        train_loss = sum(self.curr_client_losses) / len(sampled_clients)
        self.train_loss.append(train_loss)
        print("Max Lossy Client: {}, Min Loss Client: {}". format(max(self.curr_client_losses),
                                                                  min(self.curr_client_losses)))
        # Modify the gradients of malicious nodes if attack is defined
        if len(mal_nodes) > 0:
            launch_attack(attack_mode=attack_config["attack_mode"], mal_nodes=mal_nodes)
        self.agg_grad = self.aggregator.aggregate_grads(clients=sampled_clients, alphas=None)

    def update_global_model(self):
        print('server lr = {}'.format(self.lrs.get_lr()))
        dist_grads_to_model(grads=self.agg_grad, parameters=self.learner.to('cpu').parameters())
        self.opt.step()
        self.lrs.step()
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.learner.to('cpu').parameters()])
        dist_weights_to_model(weights=self.w_current, parameters=self.learner.to('cpu').parameters())

    def compute_metrics(self, curr_epoch: int, stat_freq: int = 5):
        if curr_epoch % stat_freq == 0:
            print(' ------------------------------------------ ')
            print('         Communication Round {}             '.format(curr_epoch))
            print(' -------------------------------------------')
            print('Metrics :')
            print('--------------------------------')
            print('Average Epoch Loss = {}'.format(self.train_loss[-1]))
            if self.val_loader:
                curr_val_acc, _ = infer(test_loader=self.val_loader, model=self.learner)
                self.val_acc.append(curr_val_acc)
                if curr_val_acc > self.best_val_acc:
                    self.best_val_acc = curr_val_acc
                if verbose:
                    print('Validation Acc: Curr: {} (Best: {})'.format(curr_val_acc, self.best_val_acc))
            if self.test_loader:
                curr_test_acc, _ = infer(test_loader=self.test_loader, model=self.learner)
                self.test_acc.append(curr_test_acc)
                if curr_test_acc > self.best_test_acc:
                    self.best_test_acc = curr_test_acc
                if verbose:
                    print('Test Acc: Curr: {} (Best: {})'.format(curr_test_acc, self.best_test_acc))
        print(' ')
