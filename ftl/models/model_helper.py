from .mlp import MLP
from .resnet import resnet32
from .alexnet import AlexNet
import torch
import functools


def dist_weights_to_model(weights, parameters):
    """ Given Weights and a model architecture this method updates the model
    parameters with the supplied weights """
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x*y, param.shape)
        current_data = weights[offset:offset + new_size]
        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def dist_grads_to_model(grads, parameters):
    """ Given Gradients and a model architecture this method updates the model
        gradients (Corresponding to each param) with the supplied grads """
    offset = 0

    for param in parameters:
        new_size = functools.reduce(lambda x, y: x*y, param.shape)
        current_data = grads[offset:offset + new_size]
        param.grad = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


def get_model(learner_config, data_set):
    # Load MLP
    net = learner_config["net"]
    if net == 'mlp':
        model = MLP(dim_in=learner_config.get("dim_in", 784),
                    dim_hidden1=learner_config.get("dim_hidden1", 300),
                    dim_hidden2=learner_config.get("dim_hidden2", 150),
                    drop_p=learner_config["drop_p"])
    elif net == 'alexnet':
        model = AlexNet(num_classes=learner_config.get("num_labels", 10))
    # Load ResNet 18
    elif net == 'resnet32':
        if data_set not in ['cifar10']:
            print('Resnet is not yet supported for {}'.format(data_set))
            raise NotImplementedError
        model = resnet32()
    # If Not implemented yet throw error
    else:
        raise NotImplementedError

    print('Training Model: {}'.format(net))
    print('----------------------------')
    print(model)
    return model





