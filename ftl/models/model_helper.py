# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .mlp import MLP
from .resnet import resnet32
from .alexnet import AlexNet
from .lenet import LeNet
import torch
import functools
import numpy as np
from typing import Dict


def flatten_params(learner) -> np.ndarray:
    """ Given a model flatten all params and return as np array """
    return np.concatenate([w.data.cpu().numpy().flatten() for w in learner.parameters()])


def dist_weights_to_model(weights, parameters):
    """ Given Weights and a model architecture this method updates the model parameters with the supplied weights """
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x*y, param.shape)
        current_data = weights[offset:offset + new_size]
        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def dist_grads_to_model(grads, parameters):
    """ Given Gradients and a model architecture this method updates the model gradients (Corresponding to each param)
    with the supplied grads """
    offset = 0

    for param in parameters:
        new_size = functools.reduce(lambda x, y: x*y, param.shape)
        current_data = grads[offset:offset + new_size]
        param.grad = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def get_model(learner_config: Dict, data_config: Dict):
    """ wrapper to return appropriate model class """
    net = learner_config["net"]
    data_set = data_config["data_set"]

    if net == 'mlp':
        model = MLP(dim_in=learner_config.get("dim_in", 784),
                    dim_hidden1=learner_config.get("dim_hidden1", 300),
                    dim_hidden2=learner_config.get("dim_hidden2", 150),
                    drop_p=learner_config["drop_p"])
    elif net == 'alexnet':
        if data_set not in ['cifar10']:
            raise Exception('{} is not yet supported for {}'.format(net, data_set))
        model = AlexNet(num_classes=data_config.get("num_labels", 10),
                        num_channels=data_config.get("num_channels", 3))
    elif net == 'lenet':
        if data_set not in ['mnist', 'fashion_mnist']:
            raise Exception('{} is not yet supported for {}'.format(net, data_set))
        model = LeNet(num_classes=data_config.get("num_labels", 10),
                      num_channels=data_config.get("num_channels", 1))
    elif net == 'resnet32':
        if data_set not in ['cifar10']:
            raise Exception('{} is not yet supported for {}'.format(net, data_set))
        model = resnet32()
    else:
        raise NotImplementedError

    print('Training Model: {}'.format(net))
    print('----------------------------')
    print(model)
    return model





