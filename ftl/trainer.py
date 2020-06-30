import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self):
        self.epoch_losses = []

    def train(self, data, model, optimizer, epochs=1):
        model = model.to(device)
        model.train()

        for epoch in range(1, epochs+1):
            epoch_loss = 0
            for batch_idx, (x, y) in enumerate(data):
                x, y = x.float(), y
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                y_hat = model(x)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            self.epoch_losses.append(epoch_loss/(batch_idx+1))


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
