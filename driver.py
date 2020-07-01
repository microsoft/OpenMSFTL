from ftl.experiment import run_exp
from ftl.utils import pickle_it
import argparse
import os
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')

    # Data IO Related Params
    parser.add_argument('--data_set', type=str, default='mnist',
                        help='Pass data-set')
    parser.add_argument('--dev_split', type=float, default=0.1,
                        help='Provide train test split | '
                             'fraction of data used for training')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Training mini Batch Size')
    parser.add_argument('--do_sort', type=bool, default=False)

    # Network Params
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--frac_clients', type=float, default=0.5,
                        help='For SGD pick frac of clients each round')

    # Attack Params
    parser.add_argument('--frac_adv', type=float, default=0,
                        help='Specify Fraction of Adversarial Nodes')
    parser.add_argument('--attack_mode', type=str, default='byzantine',
                        help='Options: Byzantine, Backdoor')
    parser.add_argument('--attack_model', type=str, default='gaussian')

    # Defense Params
    parser.add_argument('--agg', type=str, default='fed_avg',
                        help='Specify Aggregation/ Defence Rule. '
                             'Options: fed_avg, krum, trimmed_mean, bulyan')
    parser.add_argument('--compression_operator', type=str, default='full',
                        help='Specify Aggregation Rule,'
                             'Options: full, top, rand, svd, qsgd_biased, '
                             'qsgd_unbiased, sign, dropout_biased, dropout_unbiased')
    parser.add_argument('--num_bits', type=int, default=2)
    parser.add_argument('--frac_coordinates', type=float, default=0.5)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    # Model Params
    parser.add_argument('--m', type=str, default='mlp',
                        help='specify the network architecture you want to use')
    parser.add_argument('--dim_in', type=int, default=28*28,
                        help='in dim needed only for mlp')
    parser.add_argument('--num_channels', type=int, default=1,
                        help='num of image channels')
    # Opt Params
    parser.add_argument('--opt', type=str, default='SGD',
                        help='Pass the Optimizer you want to use')
    parser.add_argument('--lr0', type=float, default=1,
                        help='Pass the initial LR you want to use')
    parser.add_argument('--lrs', type=str, default='step',
                        help='Pass the LR Scheduler you want to use')
    parser.add_argument('--reg', type=str, default=0.05,
                        help='Pass regularization co-efficient')
    parser.add_argument('--drop_p', type=float, default=0.5,
                        help='Prob dropout model weights')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--lr_decay_rate', type=int, default=10000,
                        help='check optim _get_lr')
    parser.add_argument('--lr_restart', type=int, default=100)
    # Training params
    parser.add_argument('--num_total_epoch', type=int, default=500,
                        help='Number of Global Epochs')
    parser.add_argument('--num_comm_round', type=int, default=1000,
                        help='Number of Server Client Communication Round')

    # Results Related Params
    parser.add_argument('--o', type=str, default=None, help='Pass results location')
    parser.add_argument('--n_repeat', type=int, default=1, help='Specify number of repeat runs')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    result_file = 'num_clients_' + str(args.num_clients) + '.frac_adv_' + str(args.frac_clients) +\
                  '.attack_mode_' + args.attack_mode + '.attack_model_' + args.attack_model +\
                  '.agg_' + args.agg
    if not args.o:
        directory = "results/" + args.data_set + "/" + args.m + "/"
    else:
        directory = "results/" + args.o + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    results = []
    for random_seed in np.arange(1, args.n_repeat + 1):
        args.seed = random_seed
        results.append(run_exp(args=args))

    # Dumps the results in appropriate files
    pickle_it(args, 'parameters.' + result_file, directory)
    pickle_it(results, result_file, directory)
    print('results saved in "{}"'.format(directory))














