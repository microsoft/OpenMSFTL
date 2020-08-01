from .data_manager import DataManager
from ftl.agents import Client, Server
from ftl.training_utils import cycle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Dict, List
import os

curr_dir = os.path.dirname(__file__)
root = os.path.join(curr_dir, './data/')


class FedMNIST(DataManager):
    def __init__(self, data_config: Dict, clients: List[Client], server: Server):
        DataManager.__init__(self, data_config=data_config, clients=clients, server=server)

    def download_data(self) -> [datasets, datasets]:
        """ Downloads Data and Apply appropriate Transformations """
        mean = (0.1307,)
        std = (0.3081,)
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])
        _train_dataset = datasets.MNIST(root=root, download=self.download, train=True, transform=trans)
        _test_dataset = datasets.MNIST(root=root, download=self.download, train=False, transform=trans)

        return _train_dataset, _test_dataset

    def distribute_data(self):
        _train_dataset, _test_dataset = self.download_data()
        self.server.test_loader = DataLoader(_test_dataset)

        # update data set stats
        total_train_samples = _train_dataset.data.shape[0]
        self.num_dev = int(self.dev_split * total_train_samples)
        self.num_train = total_train_samples - self.num_dev
        self.num_test = _test_dataset.data.shape[0]
        assert self.no_of_labels == len(_train_dataset.classes), 'Number of Labels of DataSet and Model Mismatch, ' \
                                                                 'fix passed arguments to change softmax dim'
        # partition data
        self._populate_data_partition_map()

        # populate server data loaders
        val_dataset = Subset(dataset=_train_dataset, indices=self.val_ix)
        self.server.val_loader = DataLoader(val_dataset.dataset)
        self.server.test_loader = DataLoader(_test_dataset)

        # populate client data loader
        for client in self.clients:
            local_dataset = Subset(dataset=_train_dataset, indices=self.data_distribution_map[client.client_id])
            client.local_train_data = DataLoader(local_dataset.dataset, shuffle=True, batch_size=self.batch_size)
            client.trainer.train_iter = iter(cycle(client.local_train_data))








