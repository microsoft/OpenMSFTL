# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class Trainer:
    def __init__(self, optimizer=None, scheduler=None):
        self.epoch_losses = []
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_iter = None

    def train(self, model):
        model = model.to(device)
        model.train()
        x, y = next(self.train_iter)
        x, y = x.float(), y
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        self.optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        # loss = torch.nn.CrossEntropyLoss(y_hat, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
            # print('client LR = {}'.format(self.scheduler.get_lr()))
        self.epoch_losses.append(loss)


def infer(test_loader, model):
    model.eval()
    model.to(device)

    correct = 0
    with torch.no_grad():
        for batch_ix, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
