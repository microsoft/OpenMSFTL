from ftl.client import Client
from ftl.models import dist_weights_to_model
from ftl.optimization import _get_lr
from ftl.aggregation import Aggregator
from typing import List
import numpy as np
import random


class Server:
    def __init__(self,
                 args,
                 model,
                 aggregation_scheme='fed_avg',
                 clients: List[Client] = None,
                 val_loader=None,
                 test_loader=None):

        self.args = args

        # Server has access to Test and Dev Data Sets to evaluate Training Process
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Server has a pointer to all clients
        self.clients = clients
        self.aggregator = Aggregator(agg_strategy=aggregation_scheme)

        # Server keeps track of model architecture and updated weights and grads at each round
        self.global_model = model
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.global_model.parameters()])
        self.client_grads = np.empty((len(self.clients), len(self.w_current)), dtype=self.w_current.dtype)

        self.current_lr = args.lr0
        self.velocity = np.zeros(self.w_current.shape, self.client_grads.dtype)

        # Containers to store metrics
        self.test_acc = []
        self.val_acc = []
        self.train_loss = []

    def init_client_models(self):
        # Loop over all clients and update the current params
        for client in self.clients:
            # client.w_init = copy.deepcopy(self.w_current)
            dist_weights_to_model(weights=self.w_current, parameters=client.learner.parameters())

    def train_client_models(self, k, epoch):
        # Sample Clients to Train this round
        sampled_clients = random.sample(population=self.clients, k=k)

        # Now we will loop through these clients and do training steps
        # Compute number of local gradient steps per communication round
        epoch_loss = 0.0

        if epoch % self.args.lr_restart == 0:
            self.current_lr = self.args.lr0/2
        self.current_lr = _get_lr(current_lr=self.current_lr,
                                  epoch=epoch,
                                  lr_decay_rate=self.args.lr_decay_rate)
        print('current lr = {}'.format(self.current_lr))

        for client in sampled_clients:
            client.client_step(lr=self.current_lr,
                               reg=self.args.reg,
                               momentum=self.args.momentum)
            # print('Client : {} loss = {}'.format(client.client_id, client.trainer.epoch_losses[-1]))
            epoch_loss += client.trainer.epoch_losses[-1]

        # Update Metrics
        self.train_loss.append(epoch_loss / len(sampled_clients))

        # aggregate client updates
        self.aggregate_client_updates(clients=sampled_clients)

    def aggregate_client_updates(self, clients):
        """
        :param clients: Takes in a set of client compute nodes to aggregate
        :return: Updates the global model in the server with the aggregated parameters of the local models
        """
        # Now aggregate gradients and get aggregated gradient
        agg_grad = self.aggregator.compute_grad(clients=clients, client_grads=self.client_grads)

        # Now update model weights
        # TODO: Kenichi to Update with Opt Step
        self.w_current += self.args.momentum * self.velocity - self.current_lr * agg_grad

        # update the model params with these weights
        dist_weights_to_model(weights=self.w_current, parameters=self.global_model.parameters())
