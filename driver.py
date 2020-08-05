from ftl.experiment import run_exp
import argparse
import os
import numpy as np
import json
from numpyencoder import NumpyEncoder


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')

    # Client Opt Params
    parser.add_argument('--server_config', type=str, default='./configs/server_config.json')
    parser.add_argument('--client_config', type=str, default='./configs/client_config.json')
    parser.add_argument('--dga_json', type=str, default=None,
                        help='JSON config file path for dynamic gradient aggregation; '
                             'see configs/dga/rl.json for an example')
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

    directory = "result_dumps/" + client_config["data_config"]["data_set"] + "/" + \
                client_config["learner_config"]["net"] + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    results = {}
    for random_seed in np.arange(1, args.n_repeat + 1):
        client_config["data_config"]["seed"] = random_seed
        results["client_config"] = client_config
        results["server_config"] = server_config
        loss, acc, sv = run_exp(client_config=client_config, server_config=server_config)
        results["loss"] = loss
        results["acc"] = acc
        results["sv"] = sv

    with open(directory + args.o, 'w+') as f:
        json.dump(results, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


if __name__ == '__main__':
    run_main()
