from ftl.misc_utils import unpickle_dir
import numpy as np
import matplotlib.pyplot as plt


def plot_results(result, label,
                 line_style=None,
                 line_width=5,
                 marker=None):
    loss_val = result[0][0]
    loss_val_np = [loss_val_i.item() for loss_val_i in loss_val]
    x = np.arange(len(loss_val_np))
    plt.plot(x, loss_val_np, label=label, linewidth=line_width, linestyle=line_style, marker=marker)


if __name__ == '__main__':
    plt.figure()
    fig = plt.gcf()
    data_set = 'mnist'

    # MNIST
    results_dir = '/mlp/'
    data = unpickle_dir(d='./results/' + data_set + results_dir)

    # result_file = 'num_clients_' + str(args.num_clients) + \
    #               '.frac_adv_' + str(args.frac_adv) + '.attack_mode_' + args.attack_mode + \
    #               '.attack_model_' + args.attack_model + '.attack_power_' + str(args.k_std) + \
    #               '.agg_' + args.agg + \
    #               '.compression_' + args.compression_operator + '.bits_' + str(args.num_bits) + \
    #               '.frac_cd_' + str(args.frac_coordinates) + '.p_' + str(args.dropout_p) + \
    #               '.c_opt_' + args.opt + '.s_opt_' + args.server_opt

    # Baseline No Attack
    plot_results(result=data['num_clients_100.frac_adv_0.0.attack_mode_byzantine.attack_model_drift.attack_power_1'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='0% Byz', line_width=4)

    # Other
    plot_results(result=data['num_clients_100.frac_adv_0.05.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='5% Byz pow=1.5', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.1.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='10% Byz pow=1.5', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.15.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='15% Byz pow=1.5', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.2.attack_mode_byzantine.attack_model_drift.attack_power_1.5'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_opt_SGD.s_opt_Adam'],
                 label='20% Byz pow=1.5', line_width=4)

    plt.title('MNIST - Convergence Plot', fontsize=14)
    plt.grid(axis='both')
    plt.xlabel('communication round')
    plt.ylabel('loss')
    plt.legend(fontsize=11)
    plt.show()
