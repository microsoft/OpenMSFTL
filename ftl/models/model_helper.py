from .mlp import MLP
from .resnet import resnet32
from .alexnet import AlexNet
from .lenet import LeNet
from .language_model import LstmLM
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


def get_model(learner_config: Dict, data_set: str):
    """ wrapper to return appropriate model class """
    net = learner_config["net"]

    if net == 'mlp':
        model = MLP(hidden_size_list = learner_config.get("hidden_size_list", [784, 300, 150]),
                    num_classes=learner_config.get("num_labels", 10),
                    drop_p=learner_config.get("drop_p", 0.2))
    elif net == 'alexnet':
        if data_set not in ['cifar10']:
            raise Exception('{} is not yet supported for {}\nUse alexnet for cifar10'.format(net, data_set))
        model = AlexNet(num_classes=learner_config.get("num_labels", 10),
                        num_channels=learner_config.get("num_channels", 3))
    elif net == 'lenet':
        if data_set not in ['mnist', 'fashion_mnist']:
            raise Exception('{} is not yet supported for {}\nUse lenet for mnist or fashion_mnist'.format(net, data_set))
        model = LeNet(num_classes=learner_config.get("num_labels", 10),
                      num_channels=learner_config.get("num_channels", 1),
                      dim_hidden1=learner_config.get("dim_hidden1", 500))
    elif net == 'resnet32':
        if data_set not in ['cifar10']:
            raise Exception('{} is not yet supported for {}\nUse resnet32 for cifar10'.format(net, data_set))
        model = resnet32()
    elif net == 'lstmlm':
        if data_set not in ['sent140']:
            raise Exception('{} is not yet supported for {}\nUse lstmlm for sent140'.format(net, data_set))
        model = LstmLM(input_dim=learner_config["input_dim"],
                       output_dim=learner_config["output_dim"],
                       lstm_config=learner_config["lstm_config"])
    else:
        raise NotImplementedError

    print('Training Model: {}'.format(net))
    print('----------------------------')
    print(model)
    return model
