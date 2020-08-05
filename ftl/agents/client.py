from ftl.training_utils.trainer import Trainer
from ftl.training_utils.optimization import SchedulingOptimization
from ftl.compression.compression import Compression
from ftl.models.model_helper import flatten_params
import numpy as np


class Client:
    def __init__(self,
                 client_id: int,
                 learner=None,
                 attack_model=None,
                 C: Compression = None,
                 mal: bool = False,
                 T: float = 1.0,
                 client_opt_config=None,
                 client_lrs_config=None):

        self.client_id = client_id
        self.mal = mal  # is it a malicious node ?
        self.attack_model = attack_model  # pass the attack model
        self.C = C
        self.T = T
        self.local_train_data = None
        self.grad = None
        self.current_weights = None
        self.learner = learner
        self.client_opt_config = client_opt_config
        self.client_lrs_config = client_lrs_config
        self.trainer = Trainer()

    def populate_optimizer(self):
        if not self.learner:
            raise Exception("You need to populate client model before initializing optimizer")
        opt = SchedulingOptimization(model=self.learner,
                                     opt_group=self.client_opt_config,
                                     lrs_group=self.client_lrs_config)
        self.trainer.optimizer = opt.optimizer
        self.trainer.scheduler = opt.lr_scheduler

    def client_step(self):
        num_batches = self.client_opt_config.get("num_batches", 1)
        if not self.current_weights:
            self.current_weights = flatten_params(learner=self.learner)
        for batch_ix in range(0, num_batches):
            self.trainer.train(model=self.learner)
        updated_model_weights = flatten_params(learner=self.learner)
        self.grad = self.current_weights - updated_model_weights
        self.current_weights = updated_model_weights

