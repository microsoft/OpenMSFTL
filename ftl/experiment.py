from ftl.data_reader import DataReader
from ftl.client import Client
from ftl.server import Server
from ftl.models import get_model
from ftl.optimization import Optimization
from ftl.trainer import Trainer, infer
import random
import numpy as np


def run_exp(args):
    np.random.seed(args.seed)
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
        server.test_acc.append(test_acc)
        print("Test Accuracy = {}".format(test_acc))

    if plot_flag:
        plt.title('MLP', fontsize=14)
        plt.legend(fontsize=11)
        plt.plot(server.train_loss)
        plt.show()

    return server.train_loss, server.test_acc
