import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import json


def plot_driver(label: str, res_file: str, plt_type: str = 'loss',
                line_width=2, marker=None, line_style=None, optima: float = 0.0):
    with open(res_file, 'rb') as f:
        result = json.load(f)
    res = result[plt_type]
    res -= optima * np.ones(len(res))
    x = np.arange(len(res)) + np.ones(len(res))
    plt.plot(x, res, label=label, linewidth=line_width, marker=marker, linestyle=line_style)


if __name__ == '__main__':
    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    o = ['c10.bs_128.le_5',
         'c10.bs_128.le_5.mal2.ag.robust_spectral',
         'c10.bs_128.le_5.mal2.rg.robust_spectral',
         'c10.bs_128.le_5.mal2.rfg.robust_spectral',
         'c10.bs_128.le_5.mal2.sgf.robust_spectral',
         'c10.bs_128.le_5.mal2.bf.robust_spectral']
    optimal = 0
    labels = ['No Attack',
              'Additive Gaussian Attack',
              'Random Gaussian Attack',
              'Fixed Gaussian Attack',
              'Sign Flip Attack',
              'Bit Flip Attack'
              ]

    plot_type = 'loss'
    for op, label in zip(o, labels):
        result_file = './result_dumps/mnist/lenet/' + op
        plot_driver(label=label, res_file=result_file, plt_type=plot_type, optima=optimal)

    # plt.title('Spectral Gradient Aggregation')
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
        plt.xlim(-1, 201)
        plt.ylim(0, 2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    if plot_type is 'loss':
        plt.ylabel('$f -f^*$', fontsize=14)
        # plt.yscale('log')
    elif plot_type is 'test_acc':
        plt.ylabel('Test Accuracy', fontsize=14)
    elif plot_type is 'val_acc':
        plt.ylabel('Val Accuracy', fontsize=14)
    elif plot_type is 'spectral':
        plt.ylabel('Singular Value', fontsize=14)
    else:
        raise NotImplementedError

    plt.legend(fontsize=11)
    plt.show()
