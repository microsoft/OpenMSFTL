# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from ftl.agents import Client, Server
from ftl.models import get_model
from ftl.compression import Compression
from ftl.attacks import get_attack
from ftl.data_manager import process_data
import copy
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import ion, show, pause, draw
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_exp(server_config, client_config):
    data_config = client_config["data_config"]
    learner_config = client_config["learner_config"]
    client_opt_config = client_config["client_opt_config"]
    client_lrs_config = client_config["client_lrs_config"]
    attack_config = client_config.get("attack_config", {})
    client_compression_config = client_config.get("client_compression_config", {})

    server_opt_config = server_config["server_opt_config"]
    server_lrs_config = server_config["server_lrs_config"]
    aggregation_config = server_config["aggregation_config"]

    writer = SummaryWriter(server_config["summary_path"])
    print('# ------------------------------------------------- #')
    print('#               Initializing Network                #')
    print('# ------------------------------------------------- #')

    # ** Set up model architecture (learner) **
    # -----------------------------------------
    print('initializing Learner')
    model_net = get_model(learner_config=learner_config,
                          data_config=data_config).to(device=device)

    print('Setting Up the Network')
    # *** Set up Client Nodes ****
    # -----------------------------
    clients = []
    num_client_nodes = client_config["num_client_nodes"]
    num_mal_clients = int(attack_config.get("frac_adv", 0) * num_client_nodes)
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
    grad_kl_div = []
    for epoch in range(1, learner_config["comm_rounds"] + 1):
        print(" ")
        print(' ------------------------------------------ ')
        print('         Communication Round {}             '.format(epoch))
        print(' -------------------------------------------')
        server.init_client_models()
        server.train_client_models(num_participating_client=num_sampled_clients, attack_config=attack_config)
        server.update_global_model()

        print('Metrics')
        print('--------------------------------')
        print("Max Lossy Client: {}, Min Loss Client: {}".format(max(server.curr_client_losses),
                                                                 min(server.curr_client_losses)))
        print('Average Epoch Loss = {} (Best: {})'.format(server.train_loss[-1], server.lowest_epoch_loss))
        server.compute_metrics(writer=None, curr_epoch=epoch, stat_freq=server_config.get('val_freq', 5))

    return server.train_loss, server.val_acc, server.test_acc, server.aggregator.gar.Sigma_tracked, \
           server.aggregator.gar.alpha_tracked, server.best_val_acc, server.best_test_acc, \
           server.lowest_epoch_loss, grad_kl_div


def plot_grads(clients, ep, G, kl_div=None, algo='tsne'):
    ion()
    show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if algo == 'tsne':
        tsne = TSNE(n_components=2, verbose=False, perplexity=(len(clients) / 10) + 1, n_iter=300)
        proj_res = tsne.fit_transform(X=G)
    elif algo == 'pca':
        pca = PCA(n_components=3)
        proj_res = pca.fit_transform(X=G)
    else:
        raise NotImplementedError
    for ix in range(0, proj_res.shape[0]):
        color = 'b'
        if clients[ix].mal:
            color = 'r'
        ax.scatter(proj_res[ix, 0], proj_res[ix, 1], proj_res[ix, 2], c=color)
    draw()
    plt.savefig(str(ep)+'.png')
    pause(0.01)
    if kl_div:
        kl_div.append(proj_res.kl_divergence_)
