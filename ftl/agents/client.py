from ftl.training_utils.trainer import Trainer
from ftl.training_utils.optimization import SchedulingOptimization
from ftl.compression.compression import Compression
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
        src_model_weights = np.concatenate([w.data.cpu().numpy().flatten() for w in self.learner.parameters()])

        # import time
        # t0 = time.time()
        for batch_ix in range(0, num_batches):
            self.trainer.train(model=self.learner)
        # print('Time to run 1 client step {}'.format(time.time()-t0))
        # compute the local gradient
        dst_model_weights = np.concatenate([w.data.cpu().numpy().flatten() for w in self.learner.parameters()])
        self.grad = src_model_weights - dst_model_weights

