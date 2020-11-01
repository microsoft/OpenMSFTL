from torch import nn
import torch
from ftl.models.model_base import ftlModelBase


class MLP(ftlModelBase):
    def __init__(self,
                 hidden_size_list,
                 num_classes=10,
                 drop_p=0.2):
        super(MLP, self).__init__()

        num_layers = len(hidden_size_list)
        layers = []
        for i in range(num_layers-1):
            fc_i = nn.Linear(hidden_size_list[i], hidden_size_list[i+1])
            nn.init.xavier_uniform_(fc_i.weight)
            layers.append(fc_i)
            layers.append(nn.Dropout(p=drop_p))
            layers.append(nn.ReLU())

        fc_out = nn.Linear(hidden_size_list[-1], num_classes)
        nn.init.xavier_uniform_(fc_out.weight)
        layers.append(fc_out)
        self.nn = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        return self.nn(x)

