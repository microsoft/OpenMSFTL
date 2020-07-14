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
    x = np.arange(len(loss_val_np))
    plt.plot(x, loss_val_np, label=label, linewidth=line_width, linestyle=line_style, marker=marker)


def plot_driver(data, params: Dict, label: str, line_width=4):
    result_file = 'num_clients_' + str(params["num_clients"]) + \
                  '.frac_adv_' + str(params["frac_adv"]) + '.attack_mode_' + params["attack_mode"] + \
                  '.attack_model_' + params["attack_model"] + '.attack_power_' + str(params["k_std"]) + \
                  '.agg_' + params["agg"] + \
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

    # Default args
    args = {"num_clients": 100,
            "frac_adv": 0.0,
            "attack_mode": 'byzantine',
            "attack_model": 'drift',
            "k_std": 1,
            "agg": 'fed_avg',
            "compression_operator": 'full',
            "num_bits": 2,
            "frac_coordinates": 0.1,
            "dropout_p": 0.1,
            "c_opt": 'SGD',
            "server_opt": 'Adam'}

    # Baseline No Attack
    plot_driver(data=data, params=args, label='0% Byz')

    # Other

    plot_results(result=data['num_clients_100.frac_adv_0.05.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='5% Byz', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.1.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='10% Byz', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.15.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='15% Byz', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.2.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='20% Byz', line_width=4)

    plt.title('MNIST - Convergence Plot', fontsize=14)
    plt.grid(axis='both')
    plt.xlabel('communication round')
    plt.ylabel('loss')
    plt.legend(fontsize=11)
    plt.show()
