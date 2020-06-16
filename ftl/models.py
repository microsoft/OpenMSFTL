from torch import nn


def get_model(args, dim_out: int):
    if args.m == 'mlp':
        model = MLP(dim_in=args.dim_in, dim_out=dim_out)
        print('Training Model: ')
        print(model)
        return model

    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=64):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)
