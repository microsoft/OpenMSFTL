from ftl.data_reader import DataReader
from ftl.nodes import Client, Server
import argparse

"""
This is an example file depicting the use of different modules of the project.
This will be updated very frequently.
"""


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')

    # Data IO Related Params
    parser.add_argument('--data_set', type=str, default='mnist',
                        help='Pass data-set')
    parser.add_argument('--dev_split', type=float, default=0.1,
                        help='Provide train test split | fraction of data used for training')
    parser.add_argument('--batch_size', type=int, default=32,
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
    print(' Setting Up the FTL Network and distributing data ')
    num_client_nodes = args.num_clients
    clients = [Client(client_id=client_id) for client_id in range(num_client_nodes)]
    server = Server()

    # ------------------------------------------------- #
    #      Get Data and Distribute among clients        #
    # ------------------------------------------------- #
    data_set = args.data_set
    batch_size = args.batch_size
    split = args.dev_split
    data_reader = DataReader(batch_size=batch_size,
                             data_set=data_set,
                             clients=clients,
                             download=True,
                             split=split)

    server.val_loader = data_reader.val_loader
    server.test_loader = data_reader.test_loader
