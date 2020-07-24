from ftl.agents.client import Client
from typing import List


class ByzAttackCoordinated:
    def __init__(self, std: float = 1.5):
        """
        This is the Base Class for Co-ordinated Byzantine attack.
        Supported Co-ordinated attacks :
        A. "A Little Is Enough: Circumventing Defenses For Distributed Learning" (NeurIPS 2019)
        :param std: specify how many standard dev away byz nodes grad mean is from true grad mean
        """
        self.std = std

    def attack(self, byz_clients: List[Client]):
        pass
