# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License
from collections import OrderedDict
import numpy as np
import gc
from ftl.training_utils.trainer import Trainer
from ftl.training_utils.optimization import SchedulingOptimization
from ftl.compression.compression import Compression
from ftl.models.model_helper import flatten_params


class Client:
    def __init__(self,
                 client_id: int,
                 attack_model=None,
                 C: Compression = None,
                 mal: bool = False,
                 client_opt_config=None,
                 client_lrs_config=None):
        self.client_id = client_id
        self.mal = mal  # is it a malicious node ?
        self.attack_model = attack_model  # pass the attack model
        self.C = C
        self.trainer = Trainer()
        self.client_opt_config = client_opt_config
        self.client_lrs_config = client_lrs_config
        """
        These clients parameters should not be accessed by anybody
        """
        self.local_train_data = None
        self.grad = None
        self.current_weights = None
        self.learner = None

    def set_model(self, learner):
        """ initialize model and optimizer """
        self.learner = learner
        opt = SchedulingOptimization(model=self.learner,
                                     opt_group=self.client_opt_config,
                                     lrs_group=self.client_lrs_config)
        self.trainer.optimizer = opt.optimizer
        self.trainer.scheduler = opt.lr_scheduler
        self.trainer.clip_val = self.client_opt_config["clip_val"]
        self.current_weights = flatten_params(learner=self.learner)

    def client_step(self):
        """ Run Client Train (opt step) for num_batches iterations """
        num_batches = self.client_opt_config.get("num_batches", 1)
        self.trainer.reset_gradient_power()
        for batch_ix in range(0, num_batches):
            self.trainer.train(model=self.learner)
        updated_model_weights = flatten_params(learner=self.learner)
        self.grad = self.current_weights - updated_model_weights
        self.current_weights = updated_model_weights

    def empty(self):
        del self.learner, self.grad, self.current_weights
        gc.collect()

    def get_stats(self):
        """
        Return (non-privacy) stats for aggregation:
          1. Sum of negative training losses over batches
          2. Gradient mean
          3. Graident variance
        notes:
        Be cautious for changing the order of the stat element
        """

        sum_loss = self.trainer.sum_loss
        vN = self.trainer.sum_grad2 - (self.trainer.sum_grad / self.trainer.counter) * self.trainer.sum_grad
        return OrderedDict([("loss", -sum_loss),
                            ("mean", self.trainer.sum_grad / self.trainer.counter),
                            ("var", vN / self.trainer.counter)])
