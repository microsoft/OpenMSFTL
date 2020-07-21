from ftl.data_reader.data_reader import DataReader
from ftl.client import Client
from ftl.server import Server
from ftl.models.model_helper import get_model
from ftl.trainer import infer, cycle
from ftl.compression import Compression
from ftl.attacks import Attack
import copy
import random
import numpy as np


def run_exp(args):
    np.random.seed(args.seed)

    # ------------------------------------------------- #
    #               Initialize Network                  #
    # ------------------------------------------------- #

    # *** Set up Client Nodes ****
    # -----------------------------

    print(' Setting Up the FTL Network and distributing data ')
    num_client_nodes = args.num_clients
    clients = [Client(client_id=client_id) for client_id in range(num_client_nodes)]

    # Get Data and Distribute among clients
    data_reader = DataReader(batch_size=args.batch_size,
                             data_set=args.data_set,
                             clients=clients,
                             download=True,
                             split=args.dev_split,
                             do_sorting=args.do_sort)

    # Set up model architecture (learner)
    model_net = get_model(args=args, dim_out=data_reader.no_of_labels)

    # Make some client nodes adversarial
    sampled_adv_clients = random.sample(population=clients, k=int(args.frac_adv * num_client_nodes))
    for client in sampled_adv_clients:
        client.attack_mode = args.attack_mode
        client.attack_model = args.attack_model

    # Copy model architecture to clients
    # Also pass instances of compression operator
    for client in clients:
        client.learner = copy.deepcopy(model_net)
        client.trainer.train_iter = iter(cycle(client.local_train_data))
        client.C = Compression(num_bits=args.num_bits,
                               compression_function=args.compression_operator,
                               dropout_p=args.dropout_p,
                               fraction_coordinates=args.frac_coordinates)

    # **** Set up Server (Master Node)  ****
    # ---------------------------------------
    server = Server(aggregation_scheme=args.agg,
                    rank=args.rank,
                    krum_frac=args.m_krum,
                    optimizer_scheme=args.server_opt,
                    server_config={"lr0": args.server_lr0,
                                   "lr_restart": args.lr_restart,
                                   "lr_schedule": args.lrs,
                                   "lr_decay": args.lr_decay},
                    clients=clients,
                    model=copy.deepcopy(model_net),
                    val_loader=data_reader.val_loader,
                    test_loader=data_reader.test_loader)

    # ------------------------------------------------- #
    #             FTL Training                          #
    # ------------------------------------------------- #
    best_val_acc = 0.0
    corr_test_acc = 0.0

    for epoch in range(1, args.num_comm_round + 1):
        print(' ------------------------------------------ ')
        print('         Communication Round {}             '.format(epoch))
        print(' -------------------------------------------')

        server.init_client_models()
        if args.attack_mode == 'byzantine':
            attacker = Attack(k=args.k_std, attack_model=args.attack_model)
        else:
            attacker = None
        server.train_client_models(k=int(args.frac_clients * num_client_nodes),
                                   client_config={'optimizer_scheme': args.opt,
                                                  'lr': args.lr0 / 2 if epoch % args.lr_restart == 0 else args.lr0,
                                                  'weight_decay': args.reg,
                                                  'momentum': args.momentum,
                                                  'num_batches': args.num_batches
                                                  },
                                   attacker=attacker)

        print('Metrics :')
        print('--------------------------------')
        print('Average Epoch Loss = {}'.format(server.train_loss[-1]))

        val_acc, _ = infer(test_loader=server.val_loader, model=server.get_global_model())
        print("Validation Accuracy = {}".format(val_acc))
        server.val_acc.append(val_acc)

        test_acc, _ = infer(test_loader=server.test_loader, model=server.get_global_model())
        server.test_acc.append(test_acc)
        print("Test Accuracy = {}".format(test_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            corr_test_acc = test_acc

        print(' * Best Val Acc So Far *', best_val_acc)
        print(' * Corresponding Test Acc *', corr_test_acc)
        print(' ')

    return server.train_loss, server.test_acc
