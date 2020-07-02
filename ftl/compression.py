import numpy as np


class Compression:
    def __init__(self, num_bits: int,
                 compression_function: str,
                 dropout_p: float,
                 fraction_coordinates: float):
        self.compression_function = compression_function
        self.num_bits = num_bits
        self.fraction_coordinates = fraction_coordinates
        self.dropout_p = dropout_p

    def compress(self, w):

        if self.compression_function == 'full':
            """ Implements no compression i.e. returns full precision i.e all co-ordinates """
            return w

        elif self.compression_function == 'top':
            """ Retains only top k highest co-ordinates (in a norm sense) sets rest to zero """
            q = np.zeros_like(w)
            k = round(self.fraction_coordinates * q.shape[0])
            indices = np.argsort(np.abs(w))[::-1][:k]
            q[indices] = w[indices]
            return q

        elif self.compression_function == 'rand':
            """ Randomly chooses k co-ordinates to retain and set remaining to zero """
            q = np.zeros_like(w)
            k = round(self.fraction_coordinates * q.shape[0])
            indices = np.random.permutation(q.shape[0])[:k]
            q[indices] = w[indices]
            return q

        elif self.compression_function == 'dropout-biased':
            q = np.zeros_like(w)
            p = self.dropout_p
            for i in range(0, q.shape[1]):
                bin_i = np.random.binomial(1, p, (q.shape[0],))
                q[:, i] = w[:, i] * bin_i
            return q

        elif self.compression_function == 'dropout-unbiased':
            q = np.zeros_like(w)
            p = self.dropout_p
            for i in range(0, q.shape[1]):
                bin_i = np.random.binomial(1, p, (q.shape[0],))
                q[:, i] = w[:, i] * bin_i
            return q / p

        elif self.compression_function == 'qsgd':
            q = np.zeros_like(w)
            bits = self.num_bits
            s = 2 ** bits
            tau = 1 + min((np.sqrt(q.shape[0])/s), (q.shape[0]/(s**2)))
            for i in range(0, q.shape[1]):
                unif_i = np.random.rand(q.shape[0],)
                x_i = w[:, i]
                q[:, i] = ((np.sign(x_i) * np.linalg.norm(x_i))/(s*tau)) * \
                          np.floor((s*np.abs(x_i)/np.linalg.norm(x_i)) + unif_i)
            return q

        else:
            raise NotImplementedError
