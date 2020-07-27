from ftl.training_utils.misc_utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker


def plot_driver(data, params: Dict, label: str, line_width=2,
                plot_type: str = 'loss', ix: int = 0, marker=None, line_style=None):
    result_file = 'num_clients_' + str(args["num_clients"]) + \
                  '.frac_adv_' + str(args["frac_adv"]) + '.attack_mode_' + args["attack_mode"] +\
                  '.attack_model_' + args["attack_model"] + '.attack_n_std_' + str(args["attack_n_std"]) + \
                  '.attack_std_' + str(args["attack_std"]) + '.noise_scale' + str(args["noise_scale"]) +\
                  '.agg_' + args["agg"] + '.rank_' + str(args["rank"]) +\
                  '.compression_' + args["compression_operator"] + '.bits_' + str(args["num_bits"]) +\
                  '.frac_cd_' + str(args["frac_coordinates"]) + '.p_' + str(args["dropout_p"]) + \
                  '.c_opt_' + args["c_opt"] + '.s_opt_' + args["server_opt"]

    result = data[result_file]
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
    # plt.figure()
    # fig = plt.gcf()
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # MNIST
    data_set = 'mnist'
    results_dir = '/mlp/'
    data = unpickle_dir(d='./results/' + data_set + results_dir)

    # Baseline args
    args = {"num_clients": 100,
            "frac_adv": 0.0,
            "attack_mode": 'un_coordinated',
            "attack_model": 'random_gaussian',
            "attack_std": 1.0,
            "noise_scale": 1.0,
            "attack_n_std": 1.0,
            "agg": 'fed_avg',
            "rank": 20,
            "compression_operator": 'full',
            "num_bits": 2,
            "frac_coordinates": 0.1,
            "dropout_p": 0.1,
            "c_opt": 'SGD',
            "server_opt": 'Adam'}

    # -----------------------------------------------

    # Example Usage 1 :::
    # --------------------------------------
    # Plot Impact of Attacks
    # ---------------------------------------
    # # Baseline No Attack
    # Specify Plot Type
    plot_type = 'acc'
    plot_driver(data=data, params=args, label='No Attack', plot_type=plot_type)
    # # Other
    args["frac_adv"] = 0.2
    args["attack_model"] = 'additive_gaussian'
    args["attack_mode"] = 'un_coordinated'
    noise_scales = [1.0, 1.5, 2.0, 2.5, 3.0]
    labels = ["scale = 1", "scale = 1.5", "scale = 2", "scale = 2.5", "scale = 3"]
    for noise_scale, label in zip(noise_scales, labels):
         args["noise_scale"] = noise_scale
         plot_driver(data=data, params=args, label=label, plot_type=plot_type)
    plt.title('20% Uncoordinated Additive Gaussian Byz Noise', fontsize=14)
    # ----------------------------------------------------------------------------------

    # Example Usage :::
    # --------------------------------------
    # Plot Convergence (loss/ acc)
    # ---------------------------------------
    # # Specify Plot Type
    # plot_type = 'acc'
    # # plot BaseLine
    # plot_driver(data=data, params=args, label='Fed Avg', plot_type=plot_type,
    #             line_width=3)
    # args["agg"] = 'fed_lr_avg'
    # for rank in [5, 10, 15, 20, 25, 30, 35, 50]:
    #     args["rank"] = rank
    #     plot_driver(data=data, params=args, label='LR GAR(' + str(rank) + ')', plot_type=plot_type,
    #                 line_width=3)
    #
    # ----------------------------------------------------------------------------------

    # Example Usage :::
    # --------------------------------------
    # Plot SV Distribution
    # ---------------------------------------
    # Specify Plot Type
    # plot_type = 'spectral'
    # plt.title('Singular Value Distribution', fontsize=14)
    #
    # args["agg"] = 'fed_lr_avg'
    # args["rank"] = 50
    # # plot_driver(data=data, params=args, label='Client: SGD, Server: Adam', plot_type=plot_type)
    # # args["c_opt"] = 'Adam'
    # for ix in [1, 20, 50, 75, 100, 150, 200]:
    #     label = 'Comm Round = ' + str(ix)
    #     plot_driver(data=data, params=args, label=label, plot_type=plot_type, ix=ix-1,
    #                 marker='*', line_width=2, line_style='--')
    #
    # ----------------------------------------------------------------------------------

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
