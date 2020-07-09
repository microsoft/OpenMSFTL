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
    plot_results(result=data['num_clients_100.frac_adv_0.attack_mode_byzantine.attack_model_drift.attack_power1'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_optSGD.s_optAdam'],
                 label='0% Byz', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.05.attack_mode_byzantine.attack_model_drift.attack_power1'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_optSGD.s_optAdam'],
                 label='5% Byz', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.1.attack_mode_byzantine.attack_model_drift.attack_power1'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_optSGD.s_optAdam'],
                 label='10% Byz', line_width=4)
    plot_results(result=data['num_clients_100.frac_adv_0.15.attack_mode_byzantine.attack_model_drift.attack_power1'
                             '.agg_fed_avg.compression_full.bits_2.frac_cd_0.1.p_0.1.c_optSGD.s_optAdam'],
                 label='15% Byz', line_width=4)

    plt.title('MNIST', fontsize=5)

    plt.legend(fontsize=11)
    plt.show()
