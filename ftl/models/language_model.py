import torch
import torch.nn as nn
import torch.nn.functional as F
from ftl.models.model_base import ftlModelBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class LstmLM(ftlModelBase):

    def __init__(self, input_dim, output_dim, lstm_config={}):
        super(LstmLM, self).__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.bidirectional = lstm_config.get("bidirectional", False)
        self.rnn = nn.LSTM(input_size=self.input_dim,
                            hidden_size=lstm_config.get("hidden_size", 128),
                            num_layers=lstm_config.get("num_layers", 1),
                            batch_first=lstm_config.get("batch_first", True),
                            dropout=lstm_config.get("dropout", 0.1),
                            bidirectional=self.bidirectional)
        self.fc = nn.Linear(lstm_config.get("hidden_size", 100), self.output_dim)
        self.criterion = F.cross_entropy  # Can use Binary class loss: nn.BCELoss()?
        # Added items for other evaluation metrics
        self.detection_threshold = 0.5
        self.metrics['Loss'] = 0
        self.metrics['Precision'] = 0
        self.metrics['Recall'] = 0
        self.metrics['F1'] = 0


    def loss(self, batch):
        # batch might be either from:
        # 1. acoustic_dataloader
        # 2. text_dataloader

        x, y = batch["x"].float(), batch["y"].long()
        self.migrate_to_gpu(x, y)
        sum_loss = self._loss_cross_entropy(x, y)

        return sum_loss / len(x) # devided by a batch size


    def _loss_cross_entropy(self, x, y):
        """
        Primarily used for training text LMs
        """
        # y = (batch, len)
        #inp = self.embedding(y)

        # inp = (batch, len, embed_dim)
        out = self.forward(x)

        # Take the only last output, collapsing the logits and
        # targets accross time and batch_size
        batch_size, _, out_dim = out.size()
        out = out[:, -1, :].view(-1, out_dim)

        # y in batched form will have memory holes,
        # so need to apply the contiguous operation
        # y = y[:, 1:].contiguous().view(-1)
        y = y.contiguous()
        summed_loss = self.criterion(out, y)

        # accuracy
        #_, predicted = torch.max(out.data, 1)
        #summed_acc = predicted.eq(y.data).sum().item()

        return summed_loss


    def _loss_mse(self, x, y):
        """
        Primarily used for training acoustic LMs
        """
        # inp = (batch, len, embed_dim)
        out = self.forward(x)

        # Taking the last state of the network after collapsing the logits and targets accross
        # time and batch_size
        batch_size, _, out_dim = out.size()
        out = out[:,-1,:].view(-1, out_dim)
        summed_loss = F.mse_loss(out, y)
        return summed_loss/batch_size


    def forward(self, inp):
        # y = (batch, time, embed_dim)
        if self.is_cuda:
            inp  = inp.cuda()
        #inp = inp.view(150, -1, 300).float()
        o, _ = self.rnn(inp)

        if self.bidirectional:
            mid = o.size()[-1] // 2
            o = o[:, :, :mid] + o[:, :, mid:]
        # return: (batch, len, output_dim)
        return self.fc(o)


    def _evaluate_one_batch(self, x, x_len, y):
        '''
        Evaluate metrics for predictions.

        :param x (torch.tensor): input tensor for neural network
        :param y (torch.tensor): label tensor
        :return:
            - loss (torch.float): loss criterio, cross-entropy loss by default
            - accuracy (float):   accuracy of predictions (sklearn)
            - precision (float):  precision of predictions (sklearn)
            - recall (float):     recall of predictions (sklearn)
            - f1 (float):         F1-score of predictions (sklearn)
        '''
        # loss
        softmax = nn.Softmax(dim=1)
        out = self.forward(x)
        batch_size, _, out_dim = out.size()
        out = out[:, -1, :].view(-1, out_dim)
        loss = self.criterion(out, y.contiguous())
        output = softmax(out)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = accuracy_score(y.cpu(), pred.cpu())
        precision = precision_score(y.cpu(), pred.cpu())
        recall = recall_score(y.cpu(), pred.cpu())
        f1 = f1_score(y.cpu(), pred.cpu())

        return loss, accuracy, precision, recall, f1


    def validation(self, test_loader):
        """
        Evaluate a model
        Note that this is the simplest classifier and has to be overridden properly

        :return: collections.OrderedDict containing metrics values
        """
        for k in self.metrics.keys(): # Initialize metric values
            self.metrics[k] = 0

        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].float()
                y = batch['y']
                self.migrate_to_gpu(x, y)
                loss, accuracy, precision, recall, f1 = self._evaluate_one_batch(x, batch['x_len'], y)
                # sum up metrics in dict
                self.metrics['Loss'] += loss.item()
                self.metrics['Accuracy'] += accuracy
                self.metrics['Precision'] += precision
                self.metrics['Recall'] += recall
                self.metrics['F1'] += f1

        # normalize all values
        for k in self.metrics.keys():
            self.metrics[k] /= len(test_loader)

        return self.metrics