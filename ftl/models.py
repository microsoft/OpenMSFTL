from torch import nn


def get_model(args, dim_out: int):
    if args.m == 'mlp':
        model = MLP(dim_in=args.dim_in, dim_out=dim_out)
        print('Training Model: ')
        print('----------------------------')
        print(model)
        return model
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=64):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(dim_in, dim_hidden)
        nn.init.xavier_uniform_(self.fc_in.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc_out = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.fc_in(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc_out(x)
        z = self.softmax(x)

        return z


