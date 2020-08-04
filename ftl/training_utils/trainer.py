import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, optimizer=None, scheduler=None):
        self.epoch_losses = []
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_iter = None
        self.reset_gradient_power()

    def train(self, model):
        print(device)
        model = model.to(device)
        model.train()
        x, y = next(self.train_iter)
        x, y = x.float(), y
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        self.optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.__accumulate_gradient_power(model)
        self.epoch_losses.append(loss)

    ############
    def reset_gradient_power(self):
        """
        Reset the gradient stats
        """
        self.sum_grad = 0.0  # mean of gradient
        self.sum_grad2 = 0.0  # power of gradient
        self.counter = 0  # no. samples

    def __accumulate_gradient_power(self, model):
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
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_ix, (data, target) in enumerate(test_loader):
            data, target = data.float().to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy, test_loss


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
