import torch.optim as optim
import json


class SchedulingOptimizater:
    """
    This class contains an optimizer and LR scheduler.

    :param opt_alg: type of a optimizer algorithm being used: 'SGD' or 'Adam'
    :type opt_alg: string
    :param opt_group: specifying parameters of the optimizer
    :type opt_group: dict
    :param lrs: name of LR scheduler class, defaults to None
    :type lrs: string
    :param lrs_group: specifying parameters of the LR scheduler
    :type lrs_group: dict
    :param params: model parameters
    :type params: class:`nn.Module`.parameters()
    :param optimizer: optimizer instance
    :type optimizer: subclass of class:`torch.optim.Optimizer'
    :param lr_scheduler: learning scheduler instance
    :type lr_scheduler: subclass of class:`optim.lr_scheduler._LRScheduler'
    """

    def __init__(self,
                 model,
                 opt_alg = 'SGD',
                 opt_group = {},
                 lrs = None,
                 lrs_group = {},
                 verbose = 0):
        """
        :param model: model instance to be optimized
        :type model: subclass of class:`nn.Module`
        :param opt_alg: name of torch optimizer class, defaults to 'SGD'
        :type opt_alg: string
        :param opt_group: specifying parameters of the optimizer
        :type opt_group: dict
        :param lrs: name of LR scheduler class, defaults to None
        :type lrs: string
        :param lrs_group: specifying parameters of the LR scheduler
        :type lrs_group: dict
        """
        self.opt_alg = opt_alg
        self.opt_group = opt_group
        self.lrs = lrs
        self.lrs_group = lrs_group
        self.params = model.parameters()
        self.optimizer = self._get_optimizer()
        self.lr_scheduler = self._get_scheduler()

        # print information
        if verbose > 0:
            print("Optimizer Info")
            print("Type: {}".format(self.opt_alg))
            print("Params: {}".format(json.dumps(self.opt_group, indent=4)))
            if self.lrs is not None:
                print("Scheduler Info")
                print("Type: {}".format(self.lrs))
                print("Params: {}".format(json.dumps(self.lrs_group, indent=4)))

    def _get_optimizer(self):
        if self.opt_alg == 'SGD':
            return optim.SGD(params=self.params,
                             lr=self.opt_group.get('lr', 0.001),
                             momentum=self.opt_group.get('momentum', 0.0),
                             weight_decay=self.opt_group.get('weight_decay', 0.0),
                             nesterov=self.opt_group.get('nesterov', False),
                             dampening=self.opt_group.get('dampening', 0.0)
                            )
        elif self.opt_alg == "Adam":
            return optim.Adam(params=self.params,
                              lr=self.opt_group.get('lr', 0.001),
                              betas=self.opt_group.get('betas', (0.9, 0.999)),
                              eps=self.opt_group.get('eps', 1e-08),
                              weight_decay=self.opt_group.get('weight_decay', 0.0),
                              amsgrad=self.opt_group.get('amsgrad', True) # default to False in pytorch
                            )
        else:
            raise NotImplementedError("Not supported opt_alg: {}".format(self.opt_alg))

    def _get_scheduler(self):
        if self.lrs == 'StepLR':
            return optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                             step_size=self.lrs_group.get('step_size', 1),
                                             gamma=self.lrs_group.get('gamma', 0.9))
        elif self.lrs is None:
            return None
        else:
            raise NotImplementedError

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
