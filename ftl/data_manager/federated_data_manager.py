# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

import os, json
from easydict import EasyDict as edict
from abc import abstractmethod  # Note that we avoid 'abc.ABC' because of Python version compatiblity currently
from ftl.agents import Client, Server
from ftl.training_utils import cycle
from ftl.data_manager.data_text.text_dataloader import TextDataLoader
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from typing import Dict, List


class DataManagerBase:
    """
    Abstract class for data manager
    """
    def __init__(self,
                 data_config: Dict,
                 clients: List[Client],
                 server: Server):
        torch.random.manual_seed(data_config.get("seed", 11))
        self.data_config = data_config
        # keep track of data distribution among clients
        self.clients = clients
        self.server = server

    @abstractmethod
    def distribute_data(self):
        """
        Assign data to server and every client for experiments
        """
        self.server.val_loader = None # DataLoader(...)
        self.server.test_loader = None # DataLoader(...)
        for client in self.clients:
            client.trainer.train_iter = None # iter(cycle(DataLoader(...)))

        raise NotImplementedError("This method needs to be implemented")


class TorchVisionDataManager(DataManagerBase):
    """
    Base Class for TorchVision Data Readers
    """

    def __init__(self,
                 data_config: Dict,
                 clients: List[Client],
                 server: Server):

        super(TorchVisionDataManager, self).__init__(data_config=data_config,
                                                     clients=clients,
                                                     server=server)

        # Data Set Properties to be populated / can be modified
        self.num_train = 0
        self.num_dev = 0
        self.data_distribution_map = {}
        self.download = self.data_config.get('download', True)

    @staticmethod
    def _get_common_data_trans(_train_dataset):
        """ Implements a simple way to compute train and test transform that usually works """
        try:
            mean = [_train_dataset.data.float().mean(axis=(0, 1, 2)) / 255]
            std = [_train_dataset.data.float().std(axis=(0, 1, 2)) / 255]
        except:
            mean = _train_dataset.data.mean(axis=(0, 1, 2)) / 255
            std = _train_dataset.data.std(axis=(0, 1, 2)) / 255

        return mean, std

    def _populate_data_partition_map(self):
        """ wrapper to Sampling data for client, server """
        data_distribution_strategy = self.data_config.get("data_distribution_strategy", 'iid')
        if data_distribution_strategy == 'iid':
            self._iid_dist()
        else:
            raise NotImplemented

    def _iid_dist(self):
        """
        Distribute the data iid into all the clients.

        Data indexes will be assigned to each client ID by setting
        self.data_distribution_map[client_id] = [10, 12, 13] # data indecies
        """
        train_indexes = np.arange(self.num_train)

        # split rest to clients for train
        num_clients = len(self.clients)
        num_samples_per_machine = self.num_train // num_clients

        for machine_ix in range(0, num_clients - 1):
            self.data_distribution_map[self.clients[machine_ix].client_id] = \
                set(np.random.choice(a=train_indexes, size=num_samples_per_machine, replace=False))
            train_indexes = list(set(train_indexes) - self.data_distribution_map[self.clients[machine_ix].client_id])
        # put the rest in the last machine
        self.data_distribution_map[self.clients[-1].client_id] = train_indexes

    def download_data(self) -> [datasets, datasets, datasets]:
        """ Downloads Data and Apply appropriate Transformations . returns train, test dataset """
        raise NotImplementedError("This method needs to be implemented")

    def distribute_data(self):
        """ Distributes Data among clients, Server accordingly. Makes ready to train-test """
        _train_dataset, _val_dataset, _test_dataset = self.download_data()

        # update data set stats
        self.num_train = len(_train_dataset)
        self.num_dev = len(_val_dataset) #_val_dataset.data.shape[0]
        print("#training samples {}".format(self.num_train))
        print("#validation samples {}".format(self.num_dev))

        # partition data
        self._populate_data_partition_map()

        def _collate_wrapper(batch):
            """
            Convert batch data to fit in our trainer style
            """
            x = torch.stack([item[0] for item in batch], dim=0)
            y = torch.LongTensor([item[1] for item in batch])
            x_len = [1 for i in x]
            return {'x': x,
                    'x_len': x_len,
                    'y': y,
                    'y_len': x_len,
                    'utt_ids' : [None for i in y],
                    'total_frames' : sum(x_len),
                    'total_frames_with_padding' : 1.0,
                    'loss_weight' : None
                }

        # populate server data loaders
        self.server.val_loader = DataLoader(_val_dataset.dataset,
                                            batch_size=self.data_config.get("infer_batch_size", 1),
                                            pin_memory=True,
                                            num_workers=self.data_config.get("val_num_workers", 0),
                                            collate_fn=_collate_wrapper)

        self.server.test_loader = DataLoader(_test_dataset,
                                             batch_size=self.data_config.get("infer_batch_size", 1),
                                             pin_memory=True,
                                             num_workers=self.data_config.get("test_num_workers", 0),
                                             collate_fn=_collate_wrapper)

        # populate client data loader
        for client in self.clients:
            local_dataset = Subset(dataset=_train_dataset,
                                   indices=self.data_distribution_map[client.client_id])
            client.local_train_data = DataLoader(local_dataset.dataset,
                                                 shuffle=True,
                                                 batch_size=client.client_opt_config.get("train_batch_size", 256),
                                                 pin_memory=True,
                                                 num_workers=self.data_config.get("train_num_workers", 0),
                                                 collate_fn=_collate_wrapper)
            client.trainer.train_iter = iter(cycle(client.local_train_data))


class JsonlDataManager(DataManagerBase):
    """
    Data Manager Class for loading data specified with JSONL files
    """

    def __init__(self,
                 data_config: Dict,
                 clients: List[Client],
                 server: Server,
                 vec_size: int):

        super(JsonlDataManager, self).__init__(data_config=data_config,
                                               clients=clients,
                                               server=server)
        self.vec_size = vec_size

    def _distribute_train_dataloader_to_client(self):
        train_data_config = self.data_config['client']['train']
        assert os.path.exists(train_data_config["list_of_train_jsonls"]) is True, "Missing a list of training JSONL in a JSON config file: {}".format(train_data_config["list_of_train_jsonls"])
        with open(train_data_config["list_of_train_jsonls"],'r') as fid:
            data_strct = json.load(fid)

        for client_id, client in enumerate(self.clients):
            user = data_strct['users'][client_id]
            input_strct = edict({'users': [user], 'user_data': {user: data_strct['user_data'][user]}, 'num_samples': [data_strct["num_samples"][client_id]]})
            print('Loading : {}-th client with name: {}'.format(client_id, user), flush=True)
            train_dataloader = TextDataLoader(
                                            data_jsonl=input_strct,
                                            vocab_dict=train_data_config['vocab_dict'],
                                            num_workers=train_data_config.get("num_workers", 0),
                                            pin_memory=train_data_config.get("pin_memory", True),
                                            max_batch_size=train_data_config["max_batch_size"],
                                            unsorted_batch=train_data_config.get("unsorted_batch", True),
                                            batch_size=train_data_config["batch_size"],
                                            vec_size=self.vec_size,
                                            clientx=0,
                                            mode="train")
            client.trainer.train_iter = iter(cycle(train_dataloader))

    def _distributed_val_dataloader_to_server(self):
        """
        Assign a validation dataset to a server
        """
        val_data_config = self.data_config['server']['val']
        val_dataloader = TextDataLoader(
                                data_jsonl=val_data_config["val_jsonl"],
                                num_workers=val_data_config.get("num_workers", 1),
                                vocab_dict = val_data_config['vocab_dict'],
                                pin_memory=val_data_config.get("pin_memory", True),
                                unsorted_batch=val_data_config.get("unsorted_batch", False),
                                batch_size=val_data_config.get('batch_size', 1),
                                vec_size=self.vec_size,
                                clientx=None,
                                mode="val")
        self.server.val_loader = val_dataloader

    def _distributed_test_dataloader_to_server(self):
        """
        Set a test dataset to a server
        """
        test_data_config = self.data_config['server']['test']
        test_dataloader = TextDataLoader(
                            data_jsonl=test_data_config["test_jsonl"],
                            vocab_dict=test_data_config['vocab_dict'],
                            num_workers=test_data_config.get("num_workers", 1),
                            pin_memory=test_data_config.get("pin_memory", True),
                            batch_size=test_data_config.get('batch_size', 1),
                            vec_size=self.vec_size,
                            clientx=None,
                            mode="test")
        self.server.test_loader = test_dataloader

    def distribute_data(self):
        """
        Distribute text data pointed with JSONL files
        """
        self._distribute_train_dataloader_to_client()
        self._distributed_val_dataloader_to_server()
        self._distributed_test_dataloader_to_server()
