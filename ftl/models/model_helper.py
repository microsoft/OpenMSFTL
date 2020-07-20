from .mlp import MLP
from .resnet import resnet18
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


def get_model(args, dim_out: int):
    # Load MLP
    if args.m == 'mlp':
        if args.data_set not in ['mnist']:
            print('MLP not yet supported for {}'.format(args.data_set))
            raise NotImplementedError
        model = MLP(dim_in=args.dim_in, dim_out=dim_out, p=args.drop_p)

    # Load ResNet 18
    elif args.m == 'resnet18':
        if args.data_set not in ['cifar10']:
            print('Resnet is not yet supported for {}'.format(args.data_set))
            raise NotImplementedError
        model = resnet18()

    # If Not implemented yet throw error
    else:
        raise NotImplementedError

    print('Training Model: ')
    print('----------------------------')
    print(model)
    return model





