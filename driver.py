from ftl.data_reader import DataReader
from ftl.client import Client
import argparse
import math

"""
This is an example file depicting the use of different modules of the project.
This will be updated very frequently.
"""


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')

    # Data IO Related Params
    parser.add_argument('--ds', type=str, default='mnist',
                        help='Pass data-set')
    parser.add_argument('--spl', type=float, default=0.1,
                        help='Provide train test split | fraction of data used for training')
    parser.add_argument('--bs', type=int, default=32,
                        help='Training mini Batch Size')

    # Network Params
    parser.add_argument('--num_clients', type=int, default=9)
    parser.add_argument('--client_bs', type=int, default=4)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    # ------------------------------------------------- #
    #      Initialize Network , Server and Clients      #
    # ------------------------------------------------- #
    print(' Setting Up the FTL Network ')
    num_client_nodes = args.num_clients
    clients = [Client(client_id=client_id) for client_id in range(num_client_nodes)]

    # ------------------------------------------------- #
    #   Get Data : Train Data Loader, Test Data Loader  #
    # ------------------------------------------------- #
    print("--- Hang Tight !! Fetching Data --- ")
    data_set = args.ds
    batch_size = args.bs
    split = args.spl

    data_reader = DataReader(batch_size=batch_size,
                             data_set=data_set,
                             clients=clients,
                             download=True,
                             split=split)

    train_loader = data_reader.train_loader
    val_loader = data_reader.val_loader
    test_loader = data_reader.test_loader

