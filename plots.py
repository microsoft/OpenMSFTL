from ftl.training_utils.misc_utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict


def plot_driver(data, params: Dict, label: str, line_width=4, plot_type: str = 'loss', ix: int = 0):
    result_file = 'num_clients_' + str(params["num_clients"]) + \
                  '.frac_adv_' + str(params["frac_adv"]) + '.attack_mode_' + params["attack_mode"] + \
                  '.attack_model_' + params["attack_model"] + '.attack_power_' + str(params["k_std"]) + \
                  '.agg_' + params["agg"] + '.rank_' + str(params["rank"]) +\
                  '.compression_' + params["compression_operator"] + '.bits_' + str(params["num_bits"]) + \
                  '.frac_cd_' + str(params["frac_coordinates"]) + '.p_' + str(params["dropout_p"]) + \
                  '.c_opt_' + params["c_opt"] + '.s_opt_' + params["server_opt"]

    result = data[result_file]
    if plot_type is 'loss':
        res = result[0][0]
    elif plot_type is 'acc':
        res = result[0][1]
    elif plot_type is 'spectral':
        res = result[0][2]
        res = res[ix]


    else:
        raise NotImplementedError
    x = np.arange(len(res))
    plt.plot(x, res, label=label, linewidth=line_width)


if __name__ == '__main__':
    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
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
            "rank": 20,
            "compression_operator": 'full',
            "num_bits": 2,
            "frac_coordinates": 0.1,
            "dropout_p": 0.1,
            "c_opt": 'SGD',
            "server_opt": 'Adam'}

    # -----------------------------------------------

    # Example Usage :::
    # --------------------------------------
    # Plot Attacks
    # ---------------------------------------
    # # Baseline No Attack
    # Specify Plot Type
    # plot_type = 'loss'
    # plot_driver(data=data, params=args, label='No Attack', plot_type=plot_type)
    # # Other
    # args["k_std"] = 1.5
    # frac_advs = [0.05, 0.1, 0.15, 0.2]
    # labels = ["5% Byz", "10% Byz", "15% Byz", "20% Byz"]
    # for frac_adv, label in zip(frac_advs, labels):
    #     args["frac_adv"] = frac_adv
    #     plot_driver(data=data, params=args, label=label, plot_type=plot_type)
    #
    # plt.title('Byz Attack with $\sigma = 1.5$', fontsize=14)
    #
    # ----------------------------------------------------------------------------------

    # Example Usage :::
    # --------------------------------------
    # Plot SV Distribution
    # ---------------------------------------
    # Specify Plot Type
    plot_type = 'spectral'
    plt.title('Singular Value Distribution', fontsize=14)

    args["agg"] = 'fed_lr_avg'
    args["rank"] = 5
    # plot_driver(data=data, params=args, label='Client: SGD, Server: Adam', plot_type=plot_type)
    # args["c_opt"] = 'Adam'
    for ix in [1, 20, 50, 75, 100, 150, 200]:
        label = 'Comm Round = ' + str(ix)
        plot_driver(data=data, params=args, label=label, plot_type=plot_type, ix=ix-1)
    #
    # ----------------------------------------------------------------------------------

    # -------------------------------
    # ** Usually No Need to Modify **
    # -------------------------------
    plt.grid(axis='both')
    plt.tick_params(labelsize=12)

    if plot_type is 'spectral':
        plt.xlabel('Singular Value', fontsize=14)
    else:
        plt.xlabel('Communication Rounds', fontsize=14)

    if plot_type is 'loss':
        plt.ylabel('Training Loss', fontsize=14)
    elif plot_type is 'acc':
        plt.ylabel('Test Accuracy', fontsize=14)
    elif plot_type is 'Spectral':
        plt.ylabel('Magnitude', fontsize=14)
    plt.legend(fontsize=11)
    plt.show()
