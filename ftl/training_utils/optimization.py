import torch.optim as optim
import json
from typing import Dict


class SchedulingOptimization:
    def __init__(self,
                 model,
                 opt_group: Dict = None,
                 lrs_group: Dict = None,
                 verbose=0):
        """
        :param model: model instance to be optimized
        :type model: subclass of class:`nn.Module`
        :param opt_alg: name of torch optimizer class, defaults to 'SGD'
        :param opt_group: specifying parameters of the optimizer
        :param lrs: name of LR scheduler class, defaults to None
        :param lrs_group: specifying parameters of the LR scheduler
        """
        if lrs_group is None:
            lrs_group = {}
        if opt_group is None:
            opt_group = {}

        self.opt_group = opt_group
        self.lrs_group = lrs_group
        self.opt_alg = self.opt_group.get('optimizer_scheme', 'SGD')
        self.lrs = self.lrs_group.get('lr_schedule', None)

        self.params = model.parameters()
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_scheduler()

        # print information
        if verbose > 0:
            print("Optimizer Info")
            print("Type: {}".format(self.opt_alg))
            print("Params: {}".format(json.dumps(self.opt_group, indent=4)))

    def _get_optimizer(self):
        if self.opt_alg == 'SGD':
            return optim.SGD(params=self.params,
                             lr=self.opt_group.get('lr0', 0.001),
                             momentum=self.opt_group.get('momentum', 0.9),
                             weight_decay=self.opt_group.get('weight_decay', 0.05),
                             nesterov=self.opt_group.get('nesterov', False),
                             dampening=self.opt_group.get('dampening', 0.0)
                            )
        elif self.opt_alg == "Adam":
            return optim.Adam(params=self.params,
                              lr=self.opt_group.get('lr', 0.001),
                              betas=self.opt_group.get('betas', (0.9, 0.999)),
                              eps=self.opt_group.get('eps', 1e-08),
                              weight_decay=self.opt_group.get('weight_decay', 0.0),
                              amsgrad=self.opt_group.get('amsgrad', True)  # default to False in pytorch
                            )
        else:
            raise NotImplementedError("Not supported opt_alg: {}".format(self.opt_alg))

    def _get_scheduler(self):
        if self.lrs == 'StepLR':
            return optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                             step_size=self.lrs_group.get('step_size', 1),
                                             gamma=self.lrs_group.get('gamma', 0.99))
        elif self.lrs == 'MultiStepLR':
            return optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                  milestones=self.lrs_group.get('milestones', [100]),
                                                  gamma=self.lrs_group.get('gamma', 0.5),
                                                  last_epoch=self.lrs_group.get('last_epoch', -1))
        elif self.lrs is None or self.lrs == 'None':
            return None
        else:
            raise NotImplementedError


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
