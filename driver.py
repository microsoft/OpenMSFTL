from ftl.experiment import run_exp
from ftl.training_utils.misc_utils import pickle_it
import argparse
import os
import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')

    # Data IO Related Params
    parser.add_argument('--data_set', type=str, default='mnist',
                        help='Pass data-set')
    parser.add_argument('--num_labels', type=int, default=10)
    parser.add_argument('--dev_split', type=float, default=0,
                        help='Provide train test split | '
                             'fraction of data used for training')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Training mini Batch Size')
    parser.add_argument('--data_dist_strategy', type=str, default='iid')
    parser.add_argument('--download', type=bool, default=True)

    # Model Params
    parser.add_argument('--m', type=str, default='mlp',
                        help='specify the network architecture you want to use')
    parser.add_argument('--pre_trained', default=True,
                        help='Some architectures like resnet support loading pre-trained weights if this is set')
    parser.add_argument('--dim_in', type=int, default=28*28,
                        help='in dim needed only for mlp')

    # Network Params
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--frac_clients', type=float, default=0.5,
                        help='randomly pick fraction of clients each round of training')

    # Attack Params
    parser.add_argument('--frac_adv', type=float, default=0.2,
                        help='Specify Fraction of Adversarial Nodes')
    parser.add_argument('--attack_mode', type=str, default='coordinated',
                        help='Options: coordinated, un_coordinated ')
    parser.add_argument('--attack_model', type=str, default='drift',
                        help='Options: drift (Co-ordinated), random_gaussian(both),'
                             'additive_gaussian(both)')
    parser.add_argument('--attack_n_std', type=float, default=1.0,
                        help='For drift attack specify how many std away to drift the grad')
    parser.add_argument('--noise_scale', type=float, default=1.0,
                        help='scale of the gaussian noise w.r.t original value')
    parser.add_argument('--attack_std', type=float, default=1.0,
                        help='For random byz attacks drawn from gaussian specify std')

    # Defense Params
    parser.add_argument('--agg', type=str, default='fed_lr_avg',
                        help='Specify Aggregation/ Defence Rule. '
                             'Options: fed_avg, krum, trimmed_mean, bulyan')
    parser.add_argument('--rank', type=int, default=None,
                        help='For LRMF SVD rank')
    parser.add_argument('--adaptive_rank_th', type=float, default=None,
                        help='For LRMF adaptive rank based on values')
    parser.add_argument('--drop_top_comp', type=bool, default=False)
    parser.add_argument('--m_krum', type=float, default=0.7,
                        help='Krum needs m=n-f so ideally we can calculate this'
                             'accurately at each round: (num_clients - num_adv)/num_clients'
                             'but for practical purposes we treat this as hyper-parameter')
    parser.add_argument('--compression_operator', type=str, default='full',
                        help='Specify Aggregation Rule,'
                             'Options: full, top, rand, svd, qsgd_biased, '
                             'qsgd_unbiased, sign, dropout_biased, dropout_unbiased')
    parser.add_argument('--num_bits', type=int, default=2)
    parser.add_argument('--frac_coordinates', type=float, default=0.1)
    parser.add_argument('--dropout_p', type=float, default=0.1)

    # Client Opt Params
    parser.add_argument('--client_opt', type=str, default='SGD',
                        help='Name of the client optimizer: "SGD" or "Adam"')
    parser.add_argument('--client_lr0', type=float, default=0.002,
                        help='Pass the initial LR you want to use for client optimizer')
    parser.add_argument('--client_reg', type=float, default=0.05,
                        help='Pass Client L2 regularization co-efficient')
    parser.add_argument('--client_momentum', type=float, default=0.9,
                        help='Momentum of Client Optimizer')
    parser.add_argument('--num_local_steps', type=int, default=5,
                        help='Number of local client steps per comm round')

    parser.add_argument('--server_opt', type=str, default='Adam',
                        help='Name of the server (dual) optimizer: "SGD" or "Adam"')
    parser.add_argument('--server_lr0', type=float, default=0.01,
                        help='Pass the initial LR for the server optimizer')

    parser.add_argument('--lrs', type=str, default='StepLR',
                        help='Pass the LR Scheduler you want to use')

    parser.add_argument('--drop_p', type=float, default=0.5,
                        help='Prob dropout model weights')

    parser.add_argument('--lr_restart', type=int, default=100)
    parser.add_argument('--lr_decay', type=int, default=5)
    parser.add_argument('--dga_json', type=str, default=None,
                        help='JSON config file path for dynamic gradient aggregation; '
                             'see configs/dga/rl.json for an example')
    # Training params
    parser.add_argument('--num_comm_round', type=int, default=100,
                        help='Number of Server Client Communication Round')


    # Results Related Params
    parser.add_argument('--o', type=str, default=None, help='Pass results location')
    parser.add_argument('--n_repeat', type=int, default=1, help='Specify number of repeat runs')
    parser.add_argument('--seed', type=int, default=1)

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
