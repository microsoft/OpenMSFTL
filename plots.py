from dec_opt.utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import itertools

# Get Optimal Values
baselines = {'mnist': 0.35247397975085026,
             'mnist_partial': 0.07954815167630427}


def plot_results(repeats, label, plot='train',
                 optima=0.0, line_style=None, line_width=5, marker=None):
    scores = []
    for result in repeats:
        loss_val = result[0] if plot == 'train' else result[1]
        loss_val = loss_val - optima
        scores += [loss_val]

    scores = np.array(scores)
    mean = np.mean(scores, axis=0)
    x = np.arange(mean.shape[0])
    UB = mean + np.std(scores, axis=0)
    LB = mean - np.std(scores, axis=0)

    plt.plot(x, mean, label=label, linewidth=line_width, linestyle=line_style, marker=marker)
    plt.fill_between(x, LB, UB, alpha=0.2, linewidth=1)


def plot_loop(data, algorithm: List[str], n_cores: List[int],
              topology: List[str], Q: List[int], consensus_lr: List[float],
              quantization_func: List[str],
              label: List[str], optima: float,
              fraction_coordinates: List[float], dropout_p: List[float], num_bits: List[int],
              linestyle=None, line_width=5, marker=None):
    # Load Hyper Parameters
    all_hyper_param = list(itertools.product(algorithm, n_cores, topology,
                                             Q, consensus_lr, quantization_func,
                                             fraction_coordinates, dropout_p, num_bits))
    # Generate Plots
    i = 0
    for hyper_param in all_hyper_param:
        result_file = 'a_' + hyper_param[0] + '.n_' + str(hyper_param[1]) + '.t_' + hyper_param[2] + \
                      '.q_' + str(hyper_param[3]) + '.lr_' + str(hyper_param[4]) + '.c_' + hyper_param[5] + \
                      '.f_' + str(hyper_param[6]) + '.p_' + str(hyper_param[7]) + '.b_' + str(hyper_param[8])
        plot_results(repeats=data[result_file], label=label[i], optima=optima,
                     line_style=linestyle, line_width=line_width, marker=marker)
        i += 1


if __name__ == '__main__':
    plt.figure()
    fig = plt.gcf()
    data_set = 'mnist_partial'
    optimal_baseline = baselines[data_set]

    # baseline - Gradient Descent Vanilla Centralized
    baselines = unpickle_dir(d='./results/baselines')
    repeats_baseline = baselines[data_set + '_gd']
    # no communication
    repeats_disconnected = baselines[data_set + '_dis']

    plt.xlabel('Number of gradient steps', fontsize=14)
    plt.ylabel('Training Sub-optimality', fontsize=14)
    plt.grid(axis='both')
    plt.tick_params(labelsize=12)

    # Specify what result runs you want to plot together
    # this is what you need to modify

    # Now run to get plots

    # =========================
    #       Figure - 1
    # =========================
    # results_dir = '/paper/Q/'
    # data = unpickle_dir(d='./results/' + data_set + results_dir)
    # plot_results(repeats=repeats_disconnected, label='Disconnected',
    #              optima=optimal_baseline, line_width=3)
    # plot_loop(data=data, n_cores=[9],
    #           algorithm=['ours'],
    #           topology=['torus'],
    #           Q=[1, 5, 10],
    #           consensus_lr=[0.9],
    #           label=['Q=1', 'Q=5', 'Q=10'],
    #           quantization_func=['top'],
    #           fraction_coordinates=[0.5],
    #           dropout_p=[0.5], num_bits=[2],
    #           optima=optimal_baseline, linestyle=None, line_width=3)
    # plot_results(repeats=repeats_baseline, label='Centralized',
    #              optima=optimal_baseline, line_style='dashed', line_width=3)
    # plt.title('MNIST, Consensus learning rate = 0.9', fontsize=14)

    # =========================
    #       Figure - 2
    # =========================
    # results_dir = '/paper/T/'
    # data = unpickle_dir(d='./results/' + data_set + results_dir)
    # plot_loop(data=data, n_cores=[25],
    #           algorithm=['ours'],
    #           topology=['ring', 'torus', 'fully_connected'],
    #           Q=[1],
    #           consensus_lr=[0.5],
    #           label=['ring(Q=1)', 'torus(Q=1)', 'fully-connected(Q=1)'],
    #           quantization_func=['top'],
    #           fraction_coordinates=[0.5],
    #           dropout_p=[0.1], num_bits=[2],
    #           optima=optimal_baseline, linestyle='dashed', line_width=4)
    #
    # plot_loop(data=data, n_cores=[25],
    #           algorithm=['ours'],
    #           topology=['ring', 'torus', 'fully_connected'],
    #           Q=[5],
    #           consensus_lr=[0.5],
    #           label=['ring(Q=5)', 'torus(Q=5)', 'fully-connected(Q=5)'],
    #           quantization_func=['top'],
    #           fraction_coordinates=[0.5],
    #           dropout_p=[0.1], num_bits=[2],
    #           optima=optimal_baseline, linestyle=None, line_width=4)
    #
    # plot_results(repeats=repeats_baseline, label='Centralized',
    #              optima=optimal_baseline, line_width=4)
    # plt.title('MNIST, number of nodes = 25', fontsize=14)

    # =========================
    #       Figure - 3
    # =========================
    # MNIST
    results_dir = '/paper/C/'
    data = unpickle_dir(d='./results/' + data_set + results_dir)
    plot_results(repeats=data['a_ours.n_9.t_torus.q_15.lr_0.1.c_top.f_0.05.p_0.5.b_2'],
                 label='top (0.05)',
                 optima=optimal_baseline, line_width=4)
    plot_results(repeats=data['a_ours.n_9.t_torus.q_15.lr_0.05.c_rand.f_0.05.p_0.5.b_2'],
                 label='rand (0.05)',
                 optima=optimal_baseline, line_width=4)
    plot_results(repeats=data['a_ours.n_9.t_torus.q_10.lr_0.05.c_qsgd.f_0.05.p_0.5.b_2'],
                 label='qsgd (2 bit)',
                 optima=optimal_baseline, line_width=4)
    plot_results(repeats=repeats_baseline, label='exact(DGD)',
                 optima=optimal_baseline, line_style='dashed', line_width=4)
    plt.title('MNIST',  fontsize=14)

    plt.legend(fontsize=11)
    plt.yscale("log")
    plt.ylim(bottom=5e-3, top=1)
    plt.xlim(left=0, right=5000)

    plt.show()
