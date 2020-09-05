# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

import os
import json
from typing import Dict

import torch
import torch.nn as nn

from ftl.training_utils.optimization import SchedulingOptimization

import random
import numpy as np
from collections import OrderedDict


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = x.contiguous()
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True, dropout=0.0,
                 multi=1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_norm_activate = batch_norm
        self.bidirectional = bidirectional
        self.multi = multi
        self.dropout = dropout

        if self.batch_norm_activate:
            self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size))
        self.rnn = rnn_type(input_size=input_size,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            bias=True,
                            batch_first=True,
                            dropout=self.dropout)
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.batch_norm_activate:
            x = x.contiguous()
            x = self.batch_norm(x)
        x, _ = self.rnn(x)

        if self.bidirectional and self.multi < 2:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, params, wantLSTM=False, batch_norm=False):
        super(NeuralNetwork, self).__init__()

        """
        The following parameters need revisiting
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        criterion = nn.MSELoss()

        """
        self.wantLSTM = wantLSTM
        self.batch_norm = batch_norm
        layers = []

        self.softmax = nn.Softmax(dim=1)
        if self.wantLSTM:
            # Recurrent Component of the architecture
            rnns = []
            for i in range(1, len(params) - 2):
                multi = 1 if i == 1 else 1
                rnn = BatchRNN(input_size=params[i - 1] * multi,
                               hidden_size=params[i],
                               rnn_type=nn.LSTM,
                               bidirectional=True,
                               batch_norm=batch_norm,
                               multi=1,
                               dropout=0.0)
                rnns.append(('%d' % (i - 1), rnn))
            self.rnn = nn.Sequential(OrderedDict(rnns))

            layers.append(nn.Linear(params[-3], params[-2], bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(params[-2], params[-1], bias=True))
            mlp = nn.Sequential(*layers)
            self.mlp = nn.Sequential(SequenceWise(mlp), )

        else:
            if self.batch_norm:
                self.batch_norm = nn.BatchNorm1d(params[0])

            for i in range(1, len(params) - 1):
                layers.append(nn.Linear(params[i - 1], params[i], bias=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(params[-2], params[-1], bias=True))
            self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.wantLSTM:
            x = self.rnn(x)
            # x = x.squeeze(1)

        if self.batch_norm:
            x = self.batch_norm(x)
        out = self.mlp(x)
        out = out.squeeze()
        # out = self.softmax(out)
        return out


class RL:
    def __init__(self,
                 rl_config: Dict = None):
        """
        :example:
        rl_config = {
                "RL_path": "/path/to/yourfile.netw",
                "network_params": [300,128,128,128,256,64,128,100],
                "model_descriptor_RL": "Default",
                "wantLSTM": False,
                "initial_epsilon": 0.5,
                "final_epsilon": 0.0001,
                "epsilon_gamma": 0.90,
                "max_replay_memory_size": 1000,
                "minibatch_size": 16,
                "optimizer_config":{
                    "optimizer_scheme": "Adam",
                    'lrs': 'StepLR'
                    "lr": 0.0002,
                    "amsgrad": True
                },
                "annealing_config":{
                    "step_interval": "epoch",
                    "step_size": 1,
                    "gamma": 0.95
                }
        }
        """
        # Finalized config-file
        self.rl_config = rl_config
        if not 'optimizer_config' in self.rl_config:
            self.rl_config['optimizer_config'] = {'optimizer_scheme': 'Adam', 'lr': 0.0002, 'amsgrad': True,
                                                  'lrs': 'StepLR'}
        if not 'annealing_config' in self.rl_config:
            self.rl_config['annealing_config'] = {'step_interval': 'epoch', 'step_size': 1, 'gamma': 0.95}

        self.network_params = rl_config['network_params']
        if isinstance(self.network_params, str) is True:
            self.network_params = [int(x) for x in self.network_params.split(',')]

        self.out_size = self.network_params[-1]
        self.wantLSTM = rl_config.get('wantLSTM', False)
        self.max_replay_memory_size = rl_config.get('max_replay_memory_size', 1000)
        self.replay_memory = []
        self.state_memory = []
        self.epsilon = rl_config.get('initial_epsilon', 0.5)
        self.epsilon_gamma = rl_config.get('epsilon_gamma', 0.9)
        self.final_epsilon = rl_config.get('final_epsilon', 0.0001)
        self.minibatch_size = rl_config.get('minibatch_size', 16)
        self.step = 0
        self.runningLoss = 0

        model_descriptor = rl_config.get('model_descriptor_RL', 'Default')
        self.model_name = os.path.join(rl_config['RL_path'], 'rl_{}.{}.model'.format(self.out_size, model_descriptor))
        self.stats_name = os.path.join(rl_config['RL_path'], 'rl_{}.{}.stats'.format(self.out_size, model_descriptor))

        # Initialize RL model
        self.make_model()
        self.load_saved_status()

        self.criterion = nn.MSELoss()

    def forward(self, state=None):
        # epsilon greedy exploration

        if self.wantLSTM:
            N = len(state)
            state.resize(1, N)
            if len(self.state_memory) == 0:
                self.state_memory = np.zeros((self.minibatch_size, N))
            self.state_memory = np.concatenate((self.state_memory[1:], state), axis=0)
            state = self.state_memory

        if random.random() <= self.epsilon:
            print("Performed random action!")
            action = torch.rand(self.out_size).cuda() if torch.cuda.is_available() else torch.rand(self.out_size)
        else:
            state = torch.from_numpy(state).cuda() if torch.cuda.is_available() else torch.from_numpy(state)
            action = self.model(state.float())
        return action

    def train(self, batch=None):
        # save transition to replay memory
        self.replay_memory.append(batch)

        # if replay memory is full, remove the oldest transition
        if len(self.replay_memory) > self.max_replay_memory_size:
            self.replay_memory.pop(0)

        # epsilon annealing
        self.epsilon *= self.epsilon_gamma if self.epsilon * self.epsilon_gamma > self.final_epsilon else 1.0

        # sample random minibatch
        if self.wantLSTM:
            if len(self.replay_memory) >= self.minibatch_size:
                minibatch = self.replay_memory[-self.minibatch_size:]
            else:
                minibatch = self.replay_memory
        else:
            minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.minibatch_size))

        # unpack minibatch
        state_batch = torch.tensor(tuple(d[0] for d in minibatch)).float()
        action_batch = torch.tensor(tuple(d[1] for d in minibatch)).float()
        reward_batch = torch.tensor(tuple(d[2] for d in minibatch)).float().view(-1)

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = reward_batch

        # extract Q-value
        q_value = torch.sum(self.model(state_batch) * action_batch, dim=1)

        # reset gradient
        self.optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = self.criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        self.optimizer.step()

        # Tracking a running average of loss
        if self.runningLoss == 0:
            self.runningLoss = loss.item()
        else:
            self.runningLoss = 0.95 * self.runningLoss + 0.05 * loss.item()

        # Decay learning rate
        self.lr_scheduler.step()

    def make_model(self):
        # make model
        self.model = NeuralNetwork(self.network_params,
                                   self.rl_config['wantLSTM'] if 'wantLSTM' in self.rl_config else False,
                                   self.rl_config['batchNorm'] if 'batchNorm' in self.rl_config else False)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # make optimizer
        soptimizer = SchedulingOptimization(model=self.model,
                                            opt_group=self.rl_config['optimizer_config'],
                                            lrs_group=self.rl_config['annealing_config'])
        # make optimizer
        self.optimizer = soptimizer.optimizer
        # make lr_scheduler
        self.lr_scheduler = soptimizer.lr_scheduler

    def load_saved_status(self):
        if os.path.exists(self.model_name):
            print("Resuming from checkpoint model {}".format(self.model_name))
            self.load()

        if os.path.exists(self.stats_name):
            with open(self.stats_name, 'r') as logfp:  # loading the iteration no., val_loss and lr_weight
                elems = json.load(logfp)
                self.cur_iter_no = elems["i"]
                self.val_loss = elems["val_loss"]
                self.val_cer = elems["val_cer"]
                self.runningLoss = elems["weight"]

    def load(self):
        print("Loading RL checkpoint: {}".format(self.model_name))
        checkpoint = torch.load(self.model_name)

    def save(self, i, val_loss, val_cer, lr_weight=None):
        """
        Save a model as well as training information
        """
        save_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }

        outputdir = os.path.dirname(self.model_name)
        if os.path.exists(outputdir) is False:
            os.makedirs(outputdir, exist_ok=True)

        print("Saving RL model to: {}".format(self.model_name))
        try:
            torch.save(save_state, self.model_name)
        except:
            print("Failed to save {}".format(self.model_name))
            pass

        # logging the latest best values
        with open(self.stats_name, 'w') as logfp:
            json.dump({"i": i + 1,
                       "val_loss": float(val_loss),
                       "val_cer": float(val_cer),
                       "weight": float(lr_weight)},
                      logfp)
