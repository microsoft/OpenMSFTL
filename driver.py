# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from ftl.experiment import run_exp
import argparse
import os
import numpy as np
import json
from numpyencoder import NumpyEncoder
import pickle

def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')

    # Client Opt Params
    parser.add_argument('--server_config', type=str, default='./configs/server_config.json')
    parser.add_argument('--client_config', type=str, default='./configs/client_config.json')
    # Results Related Params
    parser.add_argument('--o', type=str, default='result_default', help='Pass results location')
    parser.add_argument('--n_repeat', type=int, default=1, help='Specify number of repeat runs')

    args = parser.parse_args()
    return args


def run_main():
    args = _parse_args()
    print(args)

    client_config = json.load(open(args.client_config))
    server_config = json.load(open(args.server_config))

    print('# ------------------------------------------------- #')
    print('#               Config                              #')
    print('# ------------------------------------------------- #')
    print('Server:\n{}'.format(json.dumps(server_config, indent=4)), flush=True)
    print('Client:\n{}'.format(json.dumps(client_config, indent=4)), flush=True)

    directory = "result_dumps/" + client_config["data_config"]["data_set"] + "/" + \
                client_config["learner_config"]["net"] + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    results = {}
    for random_seed in np.arange(1, args.n_repeat + 1):
        client_config["data_config"]["seed"] = random_seed
        results["client_config"] = client_config
        results["server_config"] = server_config
        loss, val_acc, test_acc, sv, alpha, best_val, best_test, lowest_loss, grad_kl_div = \
            run_exp(client_config=client_config, server_config=server_config)
        results["loss"] = loss
        results["val_acc"] = val_acc
        results["test_acc"] = test_acc
        results["sv"] = sv
        results["sv_wt"] = alpha
        results["best_val_acc"] = best_val
        results["best_test_acc"] = best_test
        results["lowest_epoch_loss"] = lowest_loss
        results["grad_kl_div"] = grad_kl_div

    print(results)
    with open(directory + args.o, 'wb') as f:
        pickle.dump(results, f)
    #    json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    run_main()
