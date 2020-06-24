from ftl.data_reader import DataReader
from ftl.client import Client
from ftl.server import Server
from ftl.models import get_model
from ftl.optimization import Optimization
from ftl.trainer import Trainer, infer

import argparse
import random


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
    parser.add_argument('--do_sort', type=bool, default=True)

    # Network Params
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--frac_clients', type=float, default=0.1,
                        help='For SGD pick frac of clients each round')
    parser.add_argument('--frac_adv', type=float, default=0,
                        help='Specify Fraction of Adversarial Nodes')

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
    parser.add_argument('--reg', type=str, default=0.1,
                        help='Pass regularization co-efficient')

    # Training params
    parser.add_argument('--num_total_epoch', type=int, default=500,
                        help='Number of Global Epochs')
    parser.add_argument('--num_comm_round', type=int, default=100,
                        help='Number of Server Client Communication Round')
    parser.add_argument('--agg', type=str, default='fed_avg',
                        help='Specify Aggregation Rule')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    try:
        import matplotlib.pyplot as plt
        plot_flag = True
    except ImportError:
        print('Disabling MatPlot Option')
        plot_flag = False

    # ------------------------------------------------- #
    #      Initialize Network , Server and Clients      #
    # ------------------------------------------------- #
    print(' Setting Up the FTL Network and distributing data ')
    num_client_nodes = args.num_clients
    clients = [Client(client_id=client_id, trainer=Trainer()) for client_id in range(num_client_nodes)]
    server = Server(aggregation=args.agg)

    # ------------------------------------------------- #
    #      Make some client nodes adversarial           #
    # ------------------------------------------------- #
    sampled_adv_clients = random.sample(population=clients, k=int(args.frac_adv * num_client_nodes))
    for client in sampled_adv_clients:
        client.attack_mode = 'byzantine'
        client.attack_model = 'gaussian'

    # ------------------------------------------------- #
    #      Get Data and Distribute among clients        #
    # ------------------------------------------------- #
    data_reader = DataReader(batch_size=args.batch_size,
                             data_set=args.data_set,
                             clients=clients,
                             download=True,
                             split=args.dev_split,
                             do_sorting=args.do_sort)

    server.val_loader = data_reader.val_loader
    server.test_loader = data_reader.test_loader

    # ------------------------------------------------- #
    #             Training Models                       #
    # ------------------------------------------------- #
    # Set up model architecture
    server.global_model = get_model(args=args, dim_out=data_reader.no_of_labels)
    # Compute number of local gradient steps per communication round
    num_local_steps = args.num_total_epoch // args.num_comm_round

    for epoch in range(1, args.num_comm_round + 1):
        print(' ------------------------------------------ ')
        print('         Communication Round {}             '. format(epoch))
        print(' -------------------------------------------')
        epoch_loss = 0.0
        # sample fraction of clients who will participate in this round
        sampled_clients = random.sample(population=clients, k=int(args.frac_clients * num_client_nodes))
        # Now loop over each client and update the local models
        for client in sampled_clients:
            client.update_local_model(model=server.global_model)
            opt = Optimization(model=client.local_model,
                               opt_alg=args.opt,
                               lr0=args.lr0,
                               reg=args.reg)
            optimizer = opt.optimizer
            lr_scheduler = opt.scheduler
            # ----------- Data Poisoning/ Backdoor -----------
            if client.attack_mode == 'backdoor':
                pass
            client.trainer.train(data=client.local_train_data,
                                 model=client.local_model,
                                 optimizer=optimizer,
                                 epochs=num_local_steps)
            lr_scheduler.step()
            print('Client : {} loss = {}'.format(client.client_id, client.trainer.epoch_losses[-1]))
            epoch_loss += client.trainer.epoch_losses[-1]

            # --------------- Byzantine ----------------------------
            # At this point check if this client is marked as byzantine
            # if its a byzantine node then we perturb the computed parameters
            # of the client node
            if client.attack_mode == 'byzantine':
                byzantine_params = client.byzantine_update(w=client.local_model.state_dict())
                client.local_model.load_state_dict(byzantine_params)

        server.train_loss.append(epoch_loss/len(sampled_clients))
        print('Metrics :')
        print('Average Epoch Loss = {}'.format(server.train_loss[-1]))
        # Now aggregate the local models and update the global models
        # so, during next epoch client local models will be updated with this aggregated model
        server.aggregate_client_updates(clients=sampled_clients)

        val_acc, _ = infer(test_loader=server.val_loader, model=server.global_model)
        print("Validation Accuracy = {}".format(val_acc))
        server.val_acc.append(val_acc)

        test_acc, _ = infer(test_loader=server.test_loader, model=server.global_model)
        server.test_acc = test_acc
        print("Test Accuracy = {}".format(test_acc))

    if plot_flag:
        plt.title('MLP', fontsize=14)
        plt.legend(fontsize=11)
        plt.plot(server.train_loss)
        plt.show()













