from ftl.training_utils.misc_utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import json


def plot_driver(label: str, result_file: str, line_width=2,
                plot_type: str = 'loss', ix: int = 0, marker=None, line_style=None):
    with open(result_file, 'rb') as f:
        result = json.load(f)
    if plot_type is 'loss':
        res = result[0][0]
    elif plot_type is 'acc':
        res = result[0][1]
    elif plot_type is 'spectral':
        res = result[0][2]
        res = res[ix]
        # Normalize singular values
        # res = res / res[0]
        # res = res / sum(res)
    else:
        raise NotImplementedError
    x = np.arange(len(res)) + np.ones(len(res))
    plt.plot(x, res, label=label, linewidth=line_width, marker=marker, linestyle=line_style)


if __name__ == '__main__':
    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    data_set = 'mnist'
    results_dir = '/mlp/'
    o = ['result_default.pickle']
    plot_type = 'loss'
    for op in o:
        result_file = './result_dumps/' + data_set + results_dir + op
        plot_driver(label="test", result_file=result_file, plot_type=plot_type)

    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    plt.grid(axis='both')
    plt.tick_params(labelsize=12)

    if plot_type is 'spectral':
        plt.xlabel('Singular Value ix', fontsize=14)
        plt.xlim(-1, 51)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    else:
        plt.xlabel('Communication Rounds', fontsize=14)
        plt.xlim(-1, 501)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

    if plot_type is 'loss':
        plt.ylabel('Training Loss', fontsize=14)
    elif plot_type is 'acc':
        plt.ylabel('Test Accuracy', fontsize=14)
    elif plot_type is 'spectral':
        plt.ylabel('Singular Value', fontsize=14)
    else:
        raise NotImplementedError

    plt.legend(fontsize=11)
    plt.show()
