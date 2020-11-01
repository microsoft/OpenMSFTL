import torch.nn as nn
import torch.nn.functional as F
import torch
from ftl.models.model_base import ftlModelBase


class LeNet(ftlModelBase):
    def __init__(self, num_channels=1, num_classes=10, dim_hidden1=500):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(4 * 4 * 50, dim_hidden1)
        self.fc_out = nn.Linear(dim_hidden1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x
