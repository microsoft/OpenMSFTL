from torch.utils.data import Dataset
import os, json, copy
import numpy as np


class ImageDataset(Dataset):
    """
    Map a Leaf's image source to the target
    """

    def __init__(self, data_jsonl, test_only, user_idx=None):

        self.utt_list=[]
        # reading the jsonl for a specific user_idx
        self.read_jsonl(data_jsonl, user_idx, test_only)
        x, _, _ = self.__getitem__(0)  # x, labels, id


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        idx = idx % len(self.utt_list)
        x = np.array(self.x[idx]['img'])
        y = np.array(self.y[idx])

        return x, y, self.user


    def read_jsonl(self, jsonl_path, user_idx, test_only):
        """
        reads in a jsonl file for a specific user (unless it's for val/testing) and returns a list of embeddings and targets.
        """
        if isinstance(jsonl_path, str):
            assert os.path.exists(jsonl_path) is True, "Missing a JSONL file for Image dataloader: {}".format(jsonl_path)
            print('Loading json-file: ', jsonl_path)
            with open(jsonl_path, 'r') as fid:
                orig_strct = json.load(fid)
        else:
            orig_strct = copy.copy(jsonl_path)
            print('Copying {}'.format(orig_strct['users']))

        self.user_list  = orig_strct['users']
        self.num_samples= orig_strct['num_samples']
        self.user_data  = orig_strct['user_data']

        if test_only:
            self.user = 'test_only'
            self.x = self.process_x(self.user_data, True)
            self.y = self.process_y(self.user_data, True)
        else:
            self.user = self.user_list[user_idx]
            self.x = self.process_x(self.user_data[self.user])
            self.y = self.process_y(self.user_data[self.user])
        assert len(self.x) == len(self.y)


    def process_x(self, raw_x_batch, test_only=False):
        img_list=[]
        if test_only:
            for user in self.user_list:
                for e in raw_x_batch[user]['x']:
                    img_list.append({'img':e, 'duration':1})
        else:
            for e in raw_x_batch['x']:
                img_list.append({'img':e, 'duration':1})

        for img in img_list:
            img["loss_weight"] = 1.0
            self.utt_list.append(img)
        return img_list


    def process_y(self, raw_y_batch, test_only=False):
        if test_only:
            y_batch = [e for i in self.user_list for e in raw_y_batch[i]['y']]
        else:
            y_batch = raw_y_batch['y']
        y_batch = np.array(y_batch)
        return y_batch

