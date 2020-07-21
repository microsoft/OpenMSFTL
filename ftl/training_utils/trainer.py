import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        self.epoch_losses = []
        self.train_iter = None

    def train(self, model, optimizer):
        model = model.to(device)
        model.train()

        x, y = next(self.train_iter)
        x, y = x.float(), y
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        loss.backward()
        self.epoch_losses.append(loss)
        optimizer.step()


def infer(test_loader, model):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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
