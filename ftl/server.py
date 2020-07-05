from ftl.client import Client
from ftl.models import dist_weights_to_model
from ftl.aggregation import Aggregator
from typing import List
import numpy as np
import random


class Server:
    def __init__(self,
                 args,
                 model,
                 aggregation_scheme='fed_avg',
                 optimizer_scheme=None,
                 clients: List[Client] = None,
                 val_loader=None,
                 test_loader=None):

        self.args = args  # TODO: Pass the minimum set of parameters instead of this lazy way

        # Server has access to Test and Dev Data Sets to evaluate Training Process
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Server has a pointer to all clients
        self.clients = clients
        self.current_lr = args.lr0
        # Aggregator tracks the model and optimizer
        self.aggregator = Aggregator(agg_strategy=aggregation_scheme,
                                     model=model,
                                     opt_alg=optimizer_scheme,
                                     opt_group={'lr': self.current_lr})

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

    def _update_lr(self, epoch):
        # TODO: This should be replaced with optim.lr_scheduler
        if epoch % self.args.lr_restart == 0:
            self.current_lr = self.args.lr0/2

        # Commented this out snce it does not work. self.current_lr = self.current_lr * args.lr_decay_rate / (epoch - 1 + args.lr_decay_rate)
        print('current lr = {}'.format(self.current_lr))

    def train_client_models(self, k, epoch):
        """
        Update each client model

        :param k: number of clients to be selected
        :param epoch: number of rounds
        """
        # Sample Clients to Train this round
        sampled_clients = random.sample(population=self.clients, k=k)

        # Now we will loop through these clients and do training steps
        # Compute number of local gradient steps per communication round
        epoch_loss = 0.0

        # update the learning rate: self.current_lr
        self._update_lr(epoch)
        for client in sampled_clients:
            client.client_step(opt_alg=self.args.opt,
                               opt_group={'lr': self.current_lr,
                                          'weight_decay': self.args.reg,
                                          'momentum': self.args.momentum}
                              )
            # print('Client : {} loss = {}'.format(client.client_id, client.trainer.epoch_losses[-1]))
            epoch_loss += client.trainer.epoch_losses[-1]

        # Update Metrics
        self.train_loss.append(epoch_loss / len(sampled_clients))

        # aggregate client updates
        self.w_current = self.aggregator.update_model(clients=sampled_clients, current_lr=self.current_lr)

