from .attack_models import DriftAttack


def get_attack(args):
    if args.attack_model == 'drift':
        return DriftAttack(n_std=args.attack_n_std)
    else:
        return None


def launch_attack(attack_mode, mal_nodes):
    if attack_mode is 'coordinated':
        # Co-ordinated Attack
        attacker = mal_nodes[0].attack_model
        print('Co-ordinated {} attack applied to {} clients'.format(mal_nodes[0].attack_model.attack_algorithm,
                                                                    len(mal_nodes)))
        attacker.attack(byz_clients=mal_nodes)

    elif attack_mode is 'un_coordinated':
        # un_coordinated stand alone attack per client
        attacker = mal_nodes[0].attack_model
        print('Un Co-ordinated {} attack applied to {} clients'.
              format(mal_nodes[0].attack_model.attack_algorithm, len(mal_nodes)))
        for mal_client in mal_nodes:
            attacker.attack(byz_clients=[mal_client])
