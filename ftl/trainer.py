import torch
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, model,
                 optimizer,
                 train_loader,
                 dev_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader

        self.iteration_losses = []
        self.epoch_losses = []

    def train(self):
        self.model = self.model.to(device)
        self.model.train()
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(self.train_loader)):
            x, y = x.float(), y
            x, y = x.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_hat = self.model(x)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            loss.backward()
            self.optimizer.step()
            self.iteration_losses.append(loss.item())
            # print("  Iteration Loss = {}".format(loss.item()))
            epoch_loss += loss.item()
        self.epoch_losses.append(epoch_loss/(batch_idx+1))

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.float().to(device), target.to(device)
                output = self.model(data)
                test_loss += torch.nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy, test_loss
