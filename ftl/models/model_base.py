from abc import abstractmethod  # Note that we avoid 'abc.ABC' because of Python version compatiblity currently
import torch
import torch.nn as nn
from collections import OrderedDict

class ftlModelBase(nn.Module):
    """
    This is the base class for every model class.
    A new model class has to be derived from this
    """
    def __init__(self, seed=1):
        torch.manual_seed(seed)
        super(ftlModelBase, self).__init__()
        # Evaluation metrics have to be defined here
        self.metrics = OrderedDict({"Accuracy": 0, "#Samples":0})

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def migrate_to_gpu(self, x, y):
        # move the inputs and target to GPU
        # if the model is also on GPU
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()

        return x, y

    @abstractmethod
    def loss(self, batch):
        """
        Implement a method to compute a loss of the model.
        """
        raise NotSupportedError("Implement loss()")

    @abstractmethod
    def validation(self, test_loader):
        """
        Implement a method to evaluate of a model
        """
        raise NotSupportedError("write validation()")


class ImageClassifierBase(ftlModelBase):

    def __init__(self, seed=1):
        super(ImageClassifierBase, self).__init__(seed=seed)

    @abstractmethod
    def loss(self, batch):
        """
        Compute a loss of the model.

        This is cross-entropy loss for the simplest classifier.
        """
        x = batch['x'].float()
        y = batch['y']
        self.migrate_to_gpu(x, y)
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)

        return loss

    @abstractmethod
    def validation(self, test_loader):
        """
        Evaluate a model

        Note that this is the simplest classifier
        """
        softmax = nn.Softmax(dim=1)
        for k in self.metrics.keys(): # Initialize metric values
            self.metrics[k] = 0

        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].float()
                y = batch['y']
                self.migrate_to_gpu(x, y)
                output = self.forward(x)
                output = softmax(output)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                self.metrics["Accuracy"] += pred.eq(y.view_as(pred)).sum().item()

        self.metrics["Accuracy"] /= len(test_loader.dataset)
        self.metrics["#Samples"] = len(test_loader.dataset)
        return self.metrics
