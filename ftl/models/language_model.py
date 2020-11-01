import torch
import torch.nn as nn
import torch.nn.functional as F
from ftl.models.model_base import ftlModelBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class RnnLM(ftlModelBase):

    def __init__(self, input_dim, output_dim, rnn_type="lstm", rnn_config={}):
        super(RnnLM, self).__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.bidirectional = rnn_config.get("bidirectional", False)
        rnn_kwargs = {"input_size": self.input_dim,
                      "hidden_size": rnn_config.get("hidden_size", 256),
                      "num_layers": rnn_config.get("num_layers", 3),
                      "batch_first": rnn_config.get("batch_first", True),
                      "dropout": rnn_config.get("dropout", 0.2),
                      "bidirectional": self.bidirectional
                    }
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(**rnn_kwargs)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(**rnn_kwargs)
        else:
            raise NotImplementedError("Unsupported RNN type {}".format(rnn_type))

        rnn_outsize = rnn_config.get("hidden_size", 256)
        if self.bidirectional is True:
            rnn_outsize *= 2
        self.fc = nn.Linear(rnn_outsize, self.output_dim)
        self.criterion = F.cross_entropy  # Can use Binary class loss: nn.BCELoss()?
        # Added items for other evaluation metrics
        self.detection_threshold = 0.5
        self.metrics['Loss'] = 0


    def loss(self, batch):
        # batch might be either from:
        # 1. acoustic_dataloader
        # 2. text_dataloader

        x, y = batch["x"].float(), batch["y"].long()
        self.migrate_to_gpu(x, y)
        sum_loss = self._loss_cross_entropy(x, batch["x_len"], y)

        return sum_loss / len(x) # devided by a batch size


    def _loss_cross_entropy(self, x, x_len, y):
        """
        Primarily used for training text LMs
        """
        # y = (batch, len)
        #inp = self.embedding(y)

        # inp = (batch, len, embed_dim)
        out = self.forward(x, x_len)

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

    def _get_target_frame(self, out, x_len):
        """
        Take the only last output if it is uni-directional.
        Otherwise, take the mid frame.
        """
        if self.bidirectional is False:
            lens = torch.LongTensor([l-1 for l in x_len])
        else:
            lens = torch.LongTensor([l//2 for l in x_len])
        # add one trailing dimension
        lens = lens.unsqueeze(-1)
        # repeat dimension
        indices = lens.repeat(1, out.size()[2])  # batch_size, _, out_dim = out.size()
        # add "empty" dimension in the middle
        indices = indices.unsqueeze(1)
        return torch.gather(out, 1, indices)


    def forward(self, inp, x_len):
        # y = (batch, time, embed_dim)
        #inp = inp.view(150, -1, 300).float()
        o, _ = self.rnn(inp)

        #if self.bidirectional:
        #    mid = o.size()[-1] // 2
        #    o = o[:, :, :mid] + o[:, :, mid:]
        # return: (batch, len, output_dim)
        o = self._get_target_frame(o, x_len)
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
        out = self.forward(x, x_len)
        batch_size, _, out_dim = out.size()
        out = out[:, -1, :].view(-1, out_dim)
        loss = self.criterion(out, y.contiguous())
        output = softmax(out)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = accuracy_score(y.cpu(), pred.cpu())

        return loss, accuracy


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
                loss, accuracy = self._evaluate_one_batch(x, batch['x_len'], y)
                # sum up metrics in dict
                self.metrics['Loss'] += loss.item()
                self.metrics['Accuracy'] += accuracy

        # normalize all values
        for k in self.metrics.keys():
            print("{}: {}/{}".format(k, self.metrics[k] / len(test_loader), self.metrics[k], len(test_loader)))
            self.metrics[k] /= len(test_loader)

        return self.metrics


class CRnnLM(RnnLM):

    def __init__(self, input_dim, output_dim, cnn_config, rnn_type="lstm", rnn_config={}):
        """

        :param cnn_config: {
                            "kernel_sizes": [[5, 300], [5, 256]],
                            "out_channels": [256, 128]
                            }
        """
        self.cnn_input = input_dim
        super(CRnnLM, self).__init__(input_dim=cnn_config["out_channels"][-1], output_dim=output_dim, rnn_type=rnn_type, rnn_config=rnn_config)
        assert self.bidirectional is False, "bidirectional RNN is not supported"

        out_channels = cnn_config.get("out_channels", [256, 128])
        kernel_sizes = cnn_config.get("kernel_sizes", [[5, 300], [5, 256]])
        paddings = cnn_config.get("paddings", [[1, 0], [0, 0]])
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels=1, out_channels=out_channels[i], kernel_size=kernel_sizes[i], padding=paddings[i])
                for i in range(len(out_channels))
            ]
        )


    def _get_target_frame(self, out, x_len):
        """
        Take the only last output if it is uni-directional.
        Otherwise, take the mid frame.
        """
        return out[:,-1,:].unsqueeze(1)


    def forward(self, x, x_len):

        for i, conv in enumerate(self.convs):
            # Since we are using conv2d, we need to add extra outer dimension
            x = x.unsqueeze(1)
            x = F.relu(conv(x)).squeeze(3)
            x = x.transpose(1, 2)

        return RnnLM.forward(self, x, x_len)
