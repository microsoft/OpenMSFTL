import torch.optim as optim


class Optimization:
    def __init__(self,
                 model,
                 lr: float,
                 momentum: float = 0.9,
                 opt_alg: str = 'SGD',
                 reg: float = 0):

        self.opt_alg = opt_alg
        self.lr = lr
        self.params = model.parameters()
        self.reg = reg
        self.momentum = momentum

        self.optimizer = self._get_optimizer()

    def _get_optimizer(self):
        if self.opt_alg == 'SGD':
            return optim.SGD(params=self.params,
                             lr=self.lr,
                             momentum=self.momentum,
                             weight_decay=self.reg)
        else:
            raise NotImplementedError


def _get_lr(current_lr, epoch, lr_decay_rate: int = 100):
    lr = current_lr * lr_decay_rate / ((epoch - 1)/10 + lr_decay_rate)
    return lr
