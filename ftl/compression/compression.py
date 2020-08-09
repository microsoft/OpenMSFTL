# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
from typing import Dict


class Compression:
    def __init__(self, compression_config: Dict):
        """
        This class applies a compression operator to the passed
        client gradient update.

        We implements the following algorithms:
        1. QSGD described in:
           Dan Alistarh et.al. QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding, NeuRips 2017
        """
        self.compression_function = compression_config.get("compression_function", 'full')
        self.num_bits = compression_config.get("num_bits", 8)
        self.fraction_coordinates = compression_config.get("fraction_coordinate", 0.5)
        self.dropout_p = compression_config.get("dropout_p", 0.5)

    def compress(self, grad, layer_wise=False):
        if layer_wise:
            raise NotImplementedError

        if self.compression_function == 'full':
            """ Implements no compression i.e. returns full precision i.e all co-ordinates """
            return grad

        elif self.compression_function == 'top':
            """ Retains only top k highest co-ordinates (in a norm sense) sets rest to zero """
            q = np.zeros_like(grad)
            k = round(self.fraction_coordinates * q.shape[0])
            indices = np.argsort(np.abs(grad))[::-1][:k]
            q[indices] = grad[indices]
            return q

        elif self.compression_function == 'rand':
            """ Randomly chooses k co-ordinates to retain and set remaining to zero """
            q = np.zeros_like(grad)
            k = round(self.fraction_coordinates * q.shape[0])
            indices = np.random.permutation(q.shape[0])[:k]
            q[indices] = grad[indices]
            return q

        elif self.compression_function == 'dropout-biased':
            """ Retain each co-ordinate with a probability p """
            q = np.zeros_like(grad)
            p = self.dropout_p
            bin_trials = np.random.binomial(1, p, (q.shape[0],))
            q = grad * bin_trials
            return q

        elif self.compression_function == 'dropout-unbiased':
            q = np.zeros_like(grad)
            p = self.dropout_p
            bin_trials = np.random.binomial(1, p, (q.shape[0],))
            q = grad * bin_trials
            return q / p

        elif self.compression_function == 'qsgd':

            raise NotImplementedError
            # q = np.zeros_like(grad)
            # bits = self.num_bits
            # s = 2 ** bits
            # tau = 1 + min((np.sqrt(q.shape[0])/s), (q.shape[0]/(s**2)))
            # for i in range(0, q.shape[1]):
            #     unif_i = np.random.rand(q.shape[0],)
            #     x_i = w[:, i]
            #     q[:, i] = ((np.sign(x_i) * np.linalg.norm(x_i))/(s*tau)) * \
            #               np.floor((s*np.abs(x_i)/np.linalg.norm(x_i)) + unif_i)
            # return q

        else:
            raise NotImplementedError
