from ftl.experiment import run_exp
from ftl.training_utils.misc_utils import pickle_it
import argparse
import os
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')

    # Client Opt Params
    parser.add_argument('--server_config', type=str, default='./configs/server_config.json')
    parser.add_argument('--client_config', type=str, default='./configs/client_config.json')
    parser.add_argument('--dga_json', type=str, default=None,
                        help='JSON config file path for dynamic gradient aggregation; '
                             'see configs/dga/rl.json for an example')
    # Results Related Params
    parser.add_argument('--o', type=str, default=None, help='Pass results location')
    parser.add_argument('--n_repeat', type=int, default=1, help='Specify number of repeat runs')

    args = parser.parse_args()
    return args
# TODO: Put configs in different config.json files grouped


def run_main():
    args = _parse_args()
    print(args)

    # TODO: Commenting FOr now
    # result_file = 'num_clients_' + str(args.num_clients) + \
    #               '.frac_adv_' + str(args.frac_adv) + '.attack_mode_' + args.attack_mode +\
    #               '.attack_model_' + args.attack_model + '.attack_n_std_' + str(args.attack_n_std) + \
    #               '.attack_std_' + str(args.attack_std) + '.noise_scale' + str(args.noise_scale) +\
    #               '.agg_' + args.agg + '.rank_' + str(args.rank) +\
    #               '.compression_' + args.compression_operator + '.bits_' + str(args.num_bits) +\
    #               '.frac_cd_' + str(args.frac_coordinates) + '.p_' + str(args.dropout_p) + \
    #               '.c_opt_' + args.opt + '.s_opt_' + args.server_opt
    #
    # if not args.o:
    #     directory = "results/" + args.data_set + "/" + args.m + "/"
    # else:
    #     directory = "results/" + args.o + '/'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    results = []
    for random_seed in np.arange(1, args.n_repeat + 1):
        args.seed = random_seed
        results.append(run_exp(args=args))

    # Commenting for now
    # Dumps the results in appropriate files
    # pickle_it(args, 'parameters.' + result_file, directory)
    # pickle_it(results, result_file, directory)
    # print('results saved in "{}"'.format(directory))


if __name__ == '__main__':
    run_main()
