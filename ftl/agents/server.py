from .client import Client
from ftl.training_utils.optimization import get_lr, SchedulingOptimization
from ftl.training_utils import infer
from ftl.models.model_helper import dist_weights_to_model, dist_grads_to_model
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
        opt_obj = SchedulingOptimization(opt_group=server_opt_config, lrs_group=server_lrs_config)
        self.opt = opt_obj.optimizer
        self.lrs = opt_obj.lr_scheduler
        self.test_acc = []
        self.val_acc = []
        self.train_loss = []

        # # Aggregator tracks the model and optimizer
        # self.aggregator = Aggregator(aggregation_config=aggregator_config,
        #                              model=server_model,
        #                              opt_group={'lr': self.current_lr, 'lrs': self.lrs})

        # # set a weight estimator for each client gradient
        # self.weight_estimator = None
        # dga_config = server_opt_config.get('dga_config', None)
        # if dga_config is not None:
        #     dga_type = dga_config.get('type', None)
        #     if dga_type == 'RL':
        #         self.weight_estimator = RLWeightEstimator(dga_config)
        #     else:
        #         raise KeyError("Invalid gradient estimator type {}".format(dga_type))

        # Server only keeps track of the pointer to the updated weights at each round
        # self.w_current = self.aggregator.w_current

    def get_global_model(self):
        """
        return: global model
        """
        return self.aggregator.model

    def init_client_models(self):
        for client in self.clients:
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
        # TODO : Parallel Calls (Defer)
        input_feature = np.zeros(3 * num_participating_client, np.float)
        # non-private stats used for weight aggregation
        for ix, client in enumerate(sampled_clients):
            client.client_step(num_batches=client_config['num_batches'])
            epoch_loss += client.trainer.epoch_losses[-1]
        # Modify the gradients of malicious nodes if attack is defined
        mal_nodes = [c for c in sampled_clients if c.mal]
        if mal_nodes:
            launch_attack(attack_mode=attack_config["attack_mode"], mal_nodes=mal_nodes)
        # now we can apply the compression operator before communicating to Server
        for client in sampled_clients:
            client.grad = client.C.compress(grad=client.grad)

        # Update Metrics
        train_loss = epoch_loss / len(sampled_clients)
        self.train_loss.append(train_loss.item())

        # update the learning rate: self.current_lr
        # self.num_rounds += 1
        # self._update_server_lr()

        # aggregate client updates
        # self._update_global_model(sampled_clients, input_feature)

    def update_global_model(self):
        agg_grad = self.aggregator.get_aggregate(clients=self.clients, alphas=None)
        self.opt.zero_grad()
        dist_grads_to_model(grads=agg_grad, parameters=self.learner.to('cpu').parameters())
        self.opt.step()
        self.lrs.step()
        self.w_current = np.concatenate([w.data.numpy().flatten() for w in self.learner.to('cpu').parameters()])

    # def _update_global_model(self, sampled_clients, input_feature):
    #     if self.weight_estimator is None:
    #         self.w_current = self.aggregator.update_model(clients=sampled_clients,
    #                                                       current_lr=self.current_lr)
    #     else:
    #         if self.weight_estimator.estimator_type == 'RL':
    #             org_aggregator_state = self.aggregator.state_dict()
    #             # run RL-based weight estimator and update a model with weighted aggregation
    #             weights = self.weight_estimator.compute_weights(input_feature)
    #             self.aggregator.update_model(clients=sampled_clients, current_lr=self.current_lr, alphas=weights)
    #             rl_aggregator_state = self.aggregator.state_dict()
    #             val_wi_rl = self.run_validation()
    #             # aggregate without the weight
    #             self.aggregator.load_state_dict(org_aggregator_state)  # revert to the original state
    #             # alphas = input_feature[0:len(sampled_clients)] / np.sum(input_feature[0:len(sampled_clients)])
    #             self.w_current = self.aggregator.update_model(clients=sampled_clients,
    #                                                           current_lr=self.current_lr, alphas=None)
    #             val_wo_rl = self.run_validation()
    #             # update the RL model
    #             should_use_rl_model = self.weight_estimator.update_model(input_feature, 1 - val_wi_rl / 100.0,
    #                                                                      1 - val_wo_rl / 100.0)
    #             if should_use_rl_model is True:  # keep the model updated with RL-based weighted aggregation
    #                 self.aggregator.load_state_dict(rl_aggregator_state)
    #                 self.w_current = self.aggregator.w_current
    #         else:
    #             raise NotImplementedError("Unsupported weight estimator type {}".
    #                                       format(self.weight_estimator.estimator_type))

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
