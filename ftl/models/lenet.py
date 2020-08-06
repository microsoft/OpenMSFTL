import torch.nn as nn
import torch.nn.functional as F
import torch
torch.manual_seed(1)


# class LeNet(nn.Module):
#     def __init__(self, num_channels=1, num_classes=10):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=20, kernel_size=5, stride=1)
#         self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
#         self.fc1 = nn.Linear(4 * 4 * 50, 500)
#         self.fc_out = nn.Linear(500, num_classes)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(x)
#         x = x.view(-1, 4 * 4 * 50)
#         x = self.fc1(x)
#         z = self.fc_out(x)
#         # z = self.softmax(x)
#         return z
class LeNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels,
                               out_channels=6,
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5)
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        z = self.fc_3(x)
        return z