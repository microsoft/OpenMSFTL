from ftl.data_reader import DataReader
from ftl.agents import Client, Server
from ftl.models import get_model
from ftl.training_utils import cycle
from ftl.comm_compression import Compression
from ftl.attacks import get_attack
import copy
import random
import json
import numpy as np


def run_exp(args):
    np.random.seed(args.seed)
    attack_config = {"frac_adv": args.frac_adv,
                     "attack_mode": args.attack_mode,
                     "attack_model": args.attack_model,
                     "attack_n_std": args.attack_n_std,
                     "noise_scale": args.noise_scale,
                     "attack_std": args.attack_std}

    server_config = {"lr0": args.server_lr0,
                     "lr_restart": args.lr_restart,
                     "lr_schedule": args.lrs,
                     "lr_decay": args.lr_decay}

    client_config = {'optimizer_scheme': args.opt,
                     'lr': args.lr0,
                     'weight_decay': args.reg,
                     'momentum': args.momentum,
                     'num_batches': args.num_batches}

    print('# ------------------------------------------------- #')
    print('#               Initialize Network                  #')
    print('# ------------------------------------------------- #')

    # *** Set up Client Nodes ****
    # -----------------------------
    print('Setting Up the FTL Network and distributing data .... ')
    num_client_nodes = args.num_clients
    clients = [Client(client_id=client_id) for client_id in range(num_client_nodes)]
    # Make some client nodes adversarial
    sampled_adv_clients = random.sample(population=clients, k=int(args.frac_adv * num_client_nodes))
    for client in sampled_adv_clients:
        client.mal = True
        client.attack_model = get_attack(attack_config=attack_config)

    # Get Data and Distribute among clients
    data_reader = DataReader(batch_size=args.batch_size,
                             data_set=args.data_set,
                             clients=clients,
                             download=True,
                             split=args.dev_split,
                             do_sorting=args.do_sort)

    # Set up model architecture (learner)
    model_net = get_model(args=args)

    num_sampled_clients = int(args.frac_clients * num_client_nodes)
    if args.dga_json is not None:
        with open(args.dga_json) as jfp:
            server_config["dga_config"] = json.load(jfp)
            assert server_config["dga_config"]["network_params"][-1] == num_sampled_clients, \
                "Invalid network output size in {}".format(args.dga_json)

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
    print("Attack config:\n{}\n".format(json.dumps(attack_config, indent=4)))
    print("Server config:\n{}\n".format(json.dumps(server_config, indent=4)))
    print("Client config:\n{}\n".format(json.dumps(client_config, indent=4)))
    server = Server(aggregation_scheme=args.agg,
                    rank=args.rank,
                    adaptive_k_th=args.adaptive_k_th,
                    krum_frac=args.m_krum,
                    optimizer_scheme=args.server_opt,
                    server_config=server_config,
                    clients=clients,
                    model=copy.deepcopy(model_net),
                    val_loader=data_reader.val_loader,
                    test_loader=data_reader.test_loader)

    print('# ------------------------------------------------- #')
    print('#            FTL Training                          #')
    print('# ------------------------------------------------- #')
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.num_comm_round + 1):
        print(' ------------------------------------------ ')
        print('         Communication Round {}             '.format(epoch))
        print(' -------------------------------------------')

        server.init_client_models()
        server.train_client_models(k=num_sampled_clients,
                                   client_config=client_config,
                                   attack_config=attack_config)

        print('Metrics :')
        print('--------------------------------')
        print('Average Epoch Loss = {}'.format(server.train_loss[-1]))

        if len(server.val_loader.dataset) > 0:
            val_acc = server.run_validation()
            print("Validation Accuracy = {}".format(val_acc))
            server.val_acc.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            print('* Best Val Acc So Far {}'.format(best_val_acc))

        if len(server.test_loader.dataset) > 0:
            test_acc = server.run_test()
            server.test_acc.append(test_acc)
            print("Test Accuracy = {}".format(test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            print('* Best Test Acc {}'.format(best_test_acc))
        print(' ')

    return server.train_loss, server.test_acc, server.aggregator.Sigma
