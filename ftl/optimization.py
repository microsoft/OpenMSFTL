import torch.optim as optim


class Optimization:
    def __init__(self, model,
                 opt_alg: str,
                 lr0: float,
                 lr_schedule: str = 'step',
                 reg: float = 0,
                 nesterov: bool = True,
                 momentum: float = 0.9,
                 dampening: float = 0):

        self.lr_schedule = lr_schedule
        self.optimizer = self.get_optimizer(opt_alg=opt_alg,
                                            lr0=lr0,
                                            params=model.parameters(),
                                            reg=reg,
                                            nesterov=nesterov,
                                            momentum=momentum,
                                            dampening=dampening)
        self.scheduler = self.get_scheduler()

    def get_optimizer(self, params, lr0, opt_alg, reg, nesterov, momentum, dampening):
        if opt_alg == 'SGD':
            return optim.SGD(params=params,
                             lr=lr0,
                             momentum=momentum,
                             weight_decay=reg,
                             nesterov=nesterov,
                             dampening=dampening)
        else:
            raise NotImplementedError

    def get_scheduler(self):
        if self.lr_schedule == 'step':
            return optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=1, gamma=0.9)
        else:
            return None
