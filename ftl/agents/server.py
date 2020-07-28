from .client import Client
from ftl.training_utils.optimization import get_lr
from ftl.training_utils import infer
from ftl.models.model_helper import dist_weights_to_model
from ftl.gradient_aggregation.weight_estimator import RLWeightEstimator
from ftl.gradient_aggregation.aggregation import Aggregator
from ftl.attacks import launch_attack
from typing import List
from typing import Dict
from torch.utils.data import DataLoader
import random
import numpy as np
import copy


class Server:
    def __init__(self,
                 model,
                 server_opt_config: Dict,
                 aggregator_config: Dict,
                 clients: List[Client] = None,
                 val_loader: DataLoader = None,
                 test_loader: DataLoader = None):

        # server opt parameters
        self.optimizer_scheme = server_opt_config["optimizer_scheme"]
        self.server_lr0 = server_opt_config["lr0"]
        self.server_lr_restart = server_opt_config["lr_restart"]
        self.lrs = server_opt_config["lr_schedule"]
        self.lr_decay = server_opt_config["lr_decay"]
        self.num_rounds = 0

        # Server has access to Test and Dev Data Sets to evaluate Training Process
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Server has a pointer to all clients
        self.clients = clients
        self.current_lr = copy.deepcopy(self.server_lr0)

        # Aggregator tracks the model and optimizer
        self.aggregator = Aggregator(aggregation_config=aggregator_config,
                                     model=model,
                                     dual_opt_alg=self.optimizer_scheme,
                                     opt_group={'lr': self.current_lr, 'lrs': self.lrs})

        # set a weight estimator for each client gradient
        self.weight_estimator = None
        dga_config = server_opt_config.get('dga_config', None)
        if dga_config is not None:
            dga_type = dga_config.get('type', None)
            if dga_type == 'RL':
                self.weight_estimator = RLWeightEstimator(dga_config)
            else:
                raise KeyError("Invalid gradient estimator type {}".format(dga_type))

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

    def train_client_models(self, k: int, client_config: Dict = None,
                            attack_config: Dict = None):
        """
        Update each client model
        :param attack_config:
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
        input_feature = np.zeros(3 * k, np.float)  # non-private stats used for weight aggregation
        for ix, client in enumerate(sampled_clients):
            client.client_step(opt_alg=client_config['optimizer_scheme'],
                               opt_group={'lr': client_config['lr'],
                                          'weight_decay': client_config['weight_decay'],
                                          'momentum': client_config['momentum']},
                               num_batches=client_config['num_batches'])
            # print('Client : {} loss = {}'.format(client.client_id, client.trainer.epoch_losses[-1]))
            # collect client's stats (normalized loss, mean of grad, var of grad)
            for sx, stat_val in enumerate(client.get_stats()):
                input_feature[ix + sx * k] = stat_val

            epoch_loss += client.trainer.epoch_losses[-1]

        # Modify the gradients of malicious nodes if attack is defined
        mal_nodes = [c for c in sampled_clients if c.mal]
        # TODO : Stop passing sampled_clients multiple times for parallelization
        if mal_nodes:
            launch_attack(attack_mode=attack_config["attack_mode"], mal_nodes=mal_nodes)

        # now we can apply the compression operator before communicating to Server
        for client in sampled_clients:
            # TODO : Stop passing sampled_clients multiple times for parallelization
            client.grad = client.C.compress(grad=client.grad)

        # Update Metrics
        train_loss = epoch_loss / len(sampled_clients)
        self.train_loss.append(train_loss.item())

        # update the learning rate: self.current_lr
        self.num_rounds += 1
        self._update_server_lr()

        # aggregate client updates
        # TODO : Stop passing sampled_clients. Compute running average of gradient if possible.
        #  This won't scale up for a larger model
        self._update_global_model(sampled_clients, input_feature)

    def _update_global_model(self, sampled_clients, input_feature):
        if self.weight_estimator is None:
            self.w_current = self.aggregator.update_model(clients=sampled_clients,
                                                          current_lr=self.current_lr)
        else:
            if self.weight_estimator.estimator_type == 'RL':
                org_aggregator_state = self.aggregator.state_dict()
                # run RL-based weight estimator and update a model with weighted aggregation
                weights = self.weight_estimator.compute_weights(input_feature)
                self.aggregator.update_model(clients=sampled_clients, current_lr=self.current_lr, alphas=weights)
                rl_aggregator_state = self.aggregator.state_dict()
                val_wi_rl = self.run_validation()
                # aggregate without the weight
                self.aggregator.load_state_dict(org_aggregator_state)  # revert to the original state
                # alphas = input_feature[0:len(sampled_clients)] / np.sum(input_feature[0:len(sampled_clients)])
                self.w_current = self.aggregator.update_model(clients=sampled_clients,
                                                              current_lr=self.current_lr, alphas=None)
                val_wo_rl = self.run_validation()
                # update the RL model
                should_use_rl_model = self.weight_estimator.update_model(input_feature, 1 - val_wi_rl / 100.0,
                                                                         1 - val_wo_rl / 100.0)
                if should_use_rl_model is True:  # keep the model updated with RL-based weighted aggregation
                    self.aggregator.load_state_dict(rl_aggregator_state)
                    self.w_current = self.aggregator.w_current
            else:
                raise NotImplementedError("Unsupported weight estimator type {}".
                                          format(self.weight_estimator.estimator_type))

    def run_validation(self):
        """
        Run validation with the current model
        """
        val_acc, _ = infer(test_loader=self.val_loader, model=self.get_global_model())
        return val_acc

    def run_test(self):
        """
        Run test with the current model
        """
        test_acc, _ = infer(test_loader=self.test_loader, model=self.get_global_model())
        return test_acc
