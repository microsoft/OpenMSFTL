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
        num_local_steps = self.args.num_total_epoch // self.args.num_comm_round
        self.current_lr = _get_lr(current_lr=self.current_lr, epoch=epoch)
        for client in sampled_clients:
            client.train_step(lr=self.current_lr,
                              reg=self.args.reg,
                              iterations=num_local_steps)
            print('Client : {} loss = {}'.format(client.client_id, client.trainer.epoch_losses[-1]))
            epoch_loss += client.trainer.epoch_losses[-1]

        # Update Metrics
        self.train_loss.append(epoch_loss / len(sampled_clients))

    def aggregate_client_updates(self, clients):
        """
        :param clients: Takes in a set of client compute nodes to aggregate
        :return: Updates the global model in the server with the aggregated parameters of the local models
        """
        # Create a container to accumulate client grads
        client_grads = np.empty((len(clients), len(self.w_current)), dtype=self.w_current.dtype)
        # Now aggregate gradients and get aggregated gradient
        agg_grad = self.aggregator.compute_grad(clients=clients, client_grads=client_grads)
        # Now update model weights
        # x_t+1 = x_t - lr * grad
        self.w_current += -self.current_lr * agg_grad
        # update the model params with these weights
        dist_weights_to_model(weights=self.w_current, parameters=self.global_model.parameters())
