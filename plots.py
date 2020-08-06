import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import json


def plot_driver(label: str, res_file: str, plt_type: str = 'loss',
                line_width=2, marker=None, line_style=None):

    with open(res_file, 'rb') as f:
        result = json.load(f)
    res = result[plt_type]
    x = np.arange(len(res)) + np.ones(len(res))
    plt.plot(x, res, label=label, linewidth=line_width, marker=marker, linestyle=line_style)


if __name__ == '__main__':
    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    o = ['fedavg.le_50.bs_128.benign', 'spavg_95.le_50.bs_128.benign']
    labels = ['FedAvg', 'SpectralAvg']
    plot_type = 'loss'
    for op, label in zip(o, labels):
        result_file = './result_dumps/mnist/lenet/' + op
        plot_driver(label=label, res_file=result_file, plt_type=plot_type)

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
        plt.xlim(-1, 31)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

    if plot_type is 'loss':
        plt.ylabel('Training Loss', fontsize=14)
        plt.yscale('log')
    elif plot_type is 'acc':
        plt.ylabel('Test Accuracy', fontsize=14)
    elif plot_type is 'spectral':
        plt.ylabel('Singular Value', fontsize=14)
    else:
        raise NotImplementedError

    plt.legend(fontsize=11)
    plt.show()
