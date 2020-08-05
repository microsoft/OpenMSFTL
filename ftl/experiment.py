from ftl.agents import Client, Server
from ftl.models import get_model
from ftl.compression import Compression
from ftl.attacks import get_attack
from ftl.data_manager import process_data
import copy
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_exp(server_config, client_config):
    data_config = client_config["data_config"]
    learner_config = client_config["learner_config"]
    client_opt_config = client_config["client_opt_config"]
    client_lrs_config = client_config["client_lrs_config"]
    attack_config = client_config["attack_config"]
    client_compression_config = client_config["client_compression_config"]

    server_opt_config = server_config["server_opt_config"]
    server_lrs_config = server_config["server_lrs_config"]
    aggregation_config = server_config["aggregation_config"]

    print('# ------------------------------------------------- #')
    print('#               Initializing Network                #')
    print('# ------------------------------------------------- #')

    # ** Set up model architecture (learner) **
    # -----------------------------------------
    print('initializing Learner')
    model_net = get_model(learner_config=learner_config,
                          data_config=data_config)

    print('Setting Up the Network')
    # *** Set up Client Nodes ****
    # -----------------------------
    clients = []
    num_client_nodes = client_config["num_client_nodes"]
    num_mal_clients = int(attack_config["frac_adv"] * num_client_nodes)
    sampled_adv_client_ix = random.sample(population=set(range(0, num_client_nodes)), k=num_mal_clients)
    for client_id in range(num_client_nodes):
        client = Client(client_id=client_id,
                        client_opt_config=client_opt_config,
                        client_lrs_config=client_lrs_config,
                        learner=copy.deepcopy(model_net).to(device=device),
                        C=Compression(compression_config=client_compression_config))
        client.populate_optimizer()
        if client_id in sampled_adv_client_ix:
            client.mal = True
            client.attack_model = get_attack(attack_config=attack_config)
        clients.append(client)

    # **** Set up Server (Master Node) ****
    # --------------------------------------
    server = Server(aggregator_config=aggregation_config,
                    server_opt_config=server_opt_config,
                    server_lrs_config=server_lrs_config,
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
    num_sampled_clients = int(client_config["fraction_participant_clients"] * num_client_nodes)
    for epoch in range(1, learner_config["comm_rounds"] + 1):
        print(' ------------------------------------------ ')
        print('         Communication Round {}             '.format(epoch))
        print(' -------------------------------------------')
        server.init_client_models()
        server.train_client_models(num_participating_client=num_sampled_clients,
                                   attack_config=attack_config)
        # Now Aggregate Gradients and Update the global model using server step
        server.update_global_model()
        print('Metrics :')
        print('--------------------------------')
        print('Average Epoch Loss = {}'.format(server.train_loss[-1]))
        server.compute_metrics(curr_epoch=epoch, stat_freq=server_config.get("verbose_freq", 5))
    return server.train_loss, server.val_acc, server.test_acc, server.aggregator.gar.Sigma_tracked, \
           server.best_val_acc, server.best_test_acc
