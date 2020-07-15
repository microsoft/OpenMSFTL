from ftl.misc_utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_results(result, label,
                 line_style=None,
                 line_width=5,
                 marker=None):
    loss_val = result[0][0]
    loss_val_np = [loss_val_i.item() for loss_val_i in loss_val]
    # optima = np.ones_like(loss_val_np) * 0.1599
    # loss_val_np -= optima
    x = np.arange(len(loss_val_np))
    plt.plot(x, loss_val_np, label=label, linewidth=line_width, linestyle=line_style, marker=marker)


def plot_driver(data, params: Dict, label: str, line_width=4):
    result_file = 'num_clients_' + str(params["num_clients"]) + \
                  '.frac_adv_' + str(params["frac_adv"]) + '.attack_mode_' + params["attack_mode"] + \
                  '.attack_model_' + params["attack_model"] + '.attack_power_' + str(params["k_std"]) + \
                  '.agg_' + params["agg"] + '.rank_' + str(params["rank"]) +\
                  '.compression_' + params["compression_operator"] + '.bits_' + str(params["num_bits"]) + \
                  '.frac_cd_' + str(params["frac_coordinates"]) + '.p_' + str(params["dropout_p"]) + \
                  '.c_opt_' + params["c_opt"] + '.s_opt_' + params["server_opt"]
    plot_results(result=data[result_file], label=label, line_width=line_width)


if __name__ == '__main__':
    plt.figure()
    fig = plt.gcf()
    data_set = 'mnist'

    # MNIST
    results_dir = '/mlp/'
    data = unpickle_dir(d='./results/' + data_set + results_dir)

    # Baseline args
    args = {"num_clients": 100,
            "frac_adv": 0.0,
            "attack_mode": 'byzantine',
            "attack_model": 'drift',
            "k_std": 1,
            "agg": 'fed_avg',
            "rank": 10,
            "compression_operator": 'full',
            "num_bits": 2,
            "frac_coordinates": 0.1,
            "dropout_p": 0.1,
            "c_opt": 'SGD',
            "server_opt": 'Adam'}

    # Plot Attacks
    # # Baseline No Attack
    # plot_driver(data=data, params=args, label='No Attack')
    # # Other
    # args["k_std"] = 1.5
    # frac_advs = [0.05, 0.1, 0.15, 0.2]
    # labels = ["5% Byz", "10% Byz", "15% Byz", "20% Byz"]
    # for frac_adv, label in zip(frac_advs, labels):
    #     args["frac_adv"] = frac_adv
    #     plot_driver(data=data, params=args, label=label)
    #
    # plt.title('Byz Attack with $\sigma = 1.5$', fontsize=14)

    # Plot LR Fed Avg
    # Baseline no SVD
    plot_driver(data=data, params=args, label='Vanilla')
    args["agg"] = 'fed_lr_avg'
    ranks = [5, 10, 15, 20, 25, 50]
    labels = ['rank=5', 'rank=10', 'rank=15', 'rank=20', 'rank=25', 'rank=50']
    for rank, label in zip(ranks, labels):
        args["rank"] = rank
        plot_driver(data=data, params=args, label=label)

    plt.title('Convergence with LR GAR', fontsize=14)
    plt.grid(axis='both')
    plt.tick_params(labelsize=12)
    plt.xlabel('Communication Rounds', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.show()
