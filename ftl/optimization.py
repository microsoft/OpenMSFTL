import torch.optim as optim


class Optimization:
    def __init__(self,
                 model,
                 lr0: float,
                 opt_alg: str = 'SGD',
                 lrs: str = 'step',
                 reg: float = 0,
                 nesterov: bool = True,
                 momentum: float = 0.9,
                 dampening: float = 0):
        self.opt_alg = opt_alg
        self.lr0 = lr0
        self.lrs = lrs
        self.params = model.parameters()
        self.reg = reg
        self.momentum = momentum
        self.nesterov = nesterov
        self.damp = dampening

        self.optimizer = self._get_optimizer()
        # self.scheduler = self._get_scheduler

    def _get_optimizer(self):
        if self.opt_alg == 'SGD':
            return optim.SGD(params=self.params,
                             lr=self.lr0,
                             momentum=self.momentum,
                             weight_decay=self.reg,
                             nesterov=self.nesterov,
                             dampening=self.damp)
        else:
            raise NotImplementedError

    # @property
    # def _get_scheduler(self):
    #     if self.lrs == 'step':
    #         return optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9)
    #     else:
    #         raise NotImplementedError


def _get_lr(current_lr, epoch, decay_rate: int = 10000):
    lr = current_lr * decay_rate / (epoch - 1 + decay_rate)
    return lr
