from torch import nn
import torch
import functools


def dist_weights_to_model(weights, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x*y, param.shape)
        current_data = weights[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def get_model(args, dim_out: int):
    if args.m == 'mlp':
        model = MLP(dim_in=args.dim_in, dim_out=dim_out, p=args.drop_p)
        print('Training Model: ')
        print('----------------------------')
        print(model)
        return model
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=100, p=0.1):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(dim_in, dim_hidden)
        nn.init.xavier_uniform_(self.fc_in.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.fc_out = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.fc_in(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.fc_out(x)
        z = self.softmax(x)

        return z


