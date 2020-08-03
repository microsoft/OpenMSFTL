from ftl.agents import Client, Server
from ftl.models import get_model
from ftl.compression import Compression
from ftl.attacks import get_attack
from ftl.data_manager import process_data
import copy
import random
import json
import numpy as np


def run_exp(args):
    np.random.seed(args.seed)

    attack_config = {
        "frac_adv": args.frac_adv,
        "attack_mode": args.attack_mode,
        "attack_model": args.attack_model,
        "attack_n_std": args.attack_n_std,
        "noise_scale": args.noise_scale,
        "attack_std": args.attack_std}

    server_opt_config = {
        "optimizer_scheme": args.server_opt,
        "lr0": args.server_lr0}

    server_lrs_config = {
        "lr_restart": args.lr_restart,
        "lr_schedule": args.lrs,
        "lr_decay": args.lr_decay
    }

    client_opt_config = {
        'optimizer_scheme': args.client_opt,
        'lr0': args.client_lr0,
        'weight_decay': args.client_reg,
        'momentum': args.client_momentum,
        "batch_size": args.batch_size,
        'num_batches': args.num_local_steps}

    client_compression_config = {
        "num_bits": args.num_bits,
        "compression_function": args.compression_operator,
        "dropout_p": args.dropout_p,
        "fraction_coordinate": args.frac_coordinates
    }

    aggregation_config = {
        "aggregation_scheme": args.agg,
        "rank": args.rank,
        "adaptive_rank_th": args.adaptive_rank_th,
        "drop_top_comp": args.drop_top_comp,
        "krum_frac": args.m_krum}

    data_config = {
        "data_set": args.data_set,
        "batch_size": args.batch_size,
        "dev_split": args.dev_split,
        "data_dist_strategy": args.data_dist_strategy,
        "seed": args.seed,
        "download": args.download,
        "num_labels": args.num_labels}

    print('# ------------------------------------------------- #')
    print('#               Initializing Network                #')
    print('# ------------------------------------------------- #')
    print("Attack config:\n{}\n".format(json.dumps(attack_config, indent=4)))
    print("Server config:\n{}\n".format(json.dumps(server_opt_config, indent=4)))
    print("Client config:\n{}\n".format(json.dumps(client_opt_config, indent=4)))
    print("Aggregation config:\n{}\n".format(json.dumps(aggregation_config, indent=4)))

    # ** Set up model architecture (learner) **
    # -----------------------------------------
    print('initializing Learner')
    model_net = get_model(args=args)

    print('Setting Up the Network')
    # *** Set up Client Nodes ****
    # -----------------------------
    clients = []
    num_client_nodes = args.num_clients
    num_mal_clients = int(attack_config["frac_adv"] * num_client_nodes)
    sampled_adv_client_ix = random.sample(population=set(range(0, num_client_nodes)), k=num_mal_clients)
    for client_id in range(num_client_nodes):
        client = Client(client_id=client_id)
        client.client_opt_config = client_opt_config
        client.learner = copy.deepcopy(model_net)
        client.C = Compression(compression_config=client_compression_config)
        client.populate_optimizer()
        if client_id in sampled_adv_client_ix:
            client.mal = True
            client.attack_model = get_attack(attack_config=attack_config)
        clients.append(client)

    # **** Set up Server (Master Node) ****
    # --------------------------------------
    server = Server(aggregator_config=aggregation_config,
                    server_opt_config=server_opt_config,
                    clients=clients,
                    server_model=copy.deepcopy(model_net),
                    val_loader=None,
                    test_loader=None)

    # ** Data Handling **
    # -------------------
    print('Processing and distributing Data across the network')
    data_manager = process_data(data_config=data_config,
                                clients=clients,
                                server=server)
    data_manager.distribute_data()

    # *** Training **
    # ---------------
    print('# ------------------------------------------------- #')
    print('#            Launching Federated Training           #')
    print('# ------------------------------------------------- #')

    num_sampled_clients = int(args.frac_clients * num_client_nodes)
    if args.dga_json is not None:
        with open(args.dga_json) as jfp:
            server_opt_config["dga_config"] = json.load(jfp)
            assert server_opt_config["dga_config"]["network_params"][-1] == num_sampled_clients, \
                "Invalid network output size in {}".format(args.dga_json)

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.num_comm_round + 1):
        print(' ------------------------------------------ ')
        print('         Communication Round {}             '.format(epoch))
        print(' -------------------------------------------')
        server.init_client_models()
        server.train_client_models(num_participating_client=num_sampled_clients,
                                   client_config=client_opt_config,
                                   attack_config=attack_config)
        # Now Aggregate Gradients and Update the global model using server step
        server.update_global_model()

        print('Metrics :')
        print('--------------------------------')
        print('Average Epoch Loss = {}'.format(server.train_loss[-1]))

        if server.val_loader:
            val_acc = server.run_validation()
            print("Validation Accuracy = {}".format(val_acc))
            server.val_acc.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            print('* Best Val Acc So Far {}'.format(best_val_acc))

        if server.test_loader:
            test_acc = server.run_test()
            server.test_acc.append(test_acc)
            print("Test Accuracy = {}".format(test_acc))
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            print('* Best Test Acc {}'.format(best_test_acc))
        print(' ')

    return server.train_loss, server.test_acc, server.aggregator.gar.Sigma_tracked
