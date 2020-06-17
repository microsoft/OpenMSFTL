from ftl.data_reader import DataReader
from ftl.nodes import Client, Server
from ftl.models import get_model
from ftl.optimization import Optimization
from ftl.trainer import Trainer, infer
from tensorboardX import SummaryWriter
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
                        help='Provide train test split | '
                             'fraction of data used for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training mini Batch Size')

    # Network Params
    parser.add_argument('--num_clients', type=int, default=9)

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
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='Pass the initial LR you want to use')
    parser.add_argument('--lrs', type=str, default='step',
                        help='Pass the LR Scheduler you want to use')
    parser.add_argument('--num_global_epoch', type=int, default=30,
                        help='Number of Global Epochs')
    parser.add_argument('--num_local_epoch', type=int, default=20,
                        help='Number of Local Epochs')

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
    clients = [Client(client_id=client_id, trainer=Trainer()) for client_id in range(num_client_nodes)]
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

    # ------------------------------------------------- #
    #             Training Models                       #
    # ------------------------------------------------- #
    # Set up model architecture
    server.global_model = get_model(args=args, dim_out=data_reader.no_of_labels)

    for epoch in range(1, args.num_global_epoch + 1):
        # Now loop over each client and update the local models
        print(' ----------------------- ')
        print('         epoch {}        '. format(epoch))
        print(' ----------------------- ')
        for client in clients:
            client.update_local_model(model=server.global_model)
            opt = Optimization(model=client.local_model,
                               opt_alg=args.opt,
                               lr0=args.lr0).get_optimizer()

            client.trainer.train(data=client.local_train_data,
                                 model=client.local_model,
                                 optimizer=opt)
            print('Client : {} loss = {}'.format(client.client_id, client.trainer.epoch_losses[-1]))

        # Now aggregate the local models and update the global models
        # so, during next epoch client local models will be updated with this aggregated model
        server.fed_average(clients=clients)
        acc, _ = infer(test_loader=server.val_loader, model=server.global_model)
        print("Validation Accuracy = {}".format(acc))











