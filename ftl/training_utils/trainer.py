# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License
import gc
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Trainer:
    def __init__(self, optimizer=None, scheduler=None, clip_val=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_iter = None
        self.clip_val = clip_val
        self.reset_gradient_power()

    @property
    def train_iter(self):
        return self.__train_iter

    @train_iter.setter
    def train_iter(self, train_data):
        self.__train_iter = iter(cycle(train_data))

    def empty(self):
        del self.__train_iter, self.scheduler, self.optimizer
        self.train_iter = None
        gc.collect()

    def train(self, model):
        model = model.to(device)
        model.train()
        batch = next(self.train_iter)
        self.optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()
        self._accumulate_gradient_power(model)
        if self.clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_val)
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            # print('client LR = {}'.format(self.scheduler.get_lr()))
        self.sum_loss += loss.detach().cpu().numpy()

    def reset_gradient_power(self):
        """
        Reset the gradient stats
        """
        self.sum_loss = 0.0
        self.sum_grad = 0.0  # mean of gradient
        self.sum_grad2 = 0.0  # power of gradient
        self.counter = 0  # no. samples

    def _accumulate_gradient_power(self, model):
        """
        Accumulate gradient stats for 1st and 2nd statistics
        """
        for p in model.parameters():
            if p.grad is None:
                continue
            p1 = torch.sum(p.grad)
            p2 = torch.sum(p.grad ** 2)
            self.sum_grad += p1
            self.sum_grad2 += p2
            self.counter += len(p.grad)

def infer(test_loader, model):
    model.to(device)
    model.eval()
    metrics = model.validation(test_loader)

    print("Results:")
    for k, v in metrics.items():
        print("{}: {}".format(k, v))

    return metrics["Accuracy"] * 100

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
