from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out=10,
                 dim_hidden1=300, dim_hidden2=100,
                 p=0.5):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(dim_in, dim_hidden1)
        nn.init.xavier_uniform_(self.fc_in.weight)
        self.fc_hidden = nn.Linear(dim_hidden1, dim_hidden2)
        nn.init.xavier_uniform_(self.fc_hidden.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)
        self.fc_out = nn.Linear(dim_hidden2, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        x = self.fc_in(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc_hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        z = self.softmax(x)

        return z