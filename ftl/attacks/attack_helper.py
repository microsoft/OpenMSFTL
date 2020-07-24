from .attack_models import DriftAttack
from typing import Dict


def get_attack(attack_config: Dict):
    if attack_config["attack_model"] == 'drift':
        return DriftAttack(attack_config= attack_config)
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
