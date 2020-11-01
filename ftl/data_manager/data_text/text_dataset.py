from torch.utils.data import Dataset
import os, json, copy
import numpy as np

from .language_utils import line_to_indices, val_to_vec


class TextDataset(Dataset):
    """
    Map a text source to the target text
    """

    def __init__(self, data_jsonl, word_emb_arr, indd, vocab, vec_size=300, test_only=False, user_idx=None, emb_arr=None):

        self.utt_list=[]

        #self.word_emb_arr, self.indd, self.vocab = get_word_emb_arr(vocab_dict)
        self.word_emb_arr = word_emb_arr
        self.indd = indd
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        if emb_arr:
            assert isinstance(emb_arr, dict)==True, 'emb_arr needs to be a dict'
            self.word_emb_arr=np.array(emb_arr[self.vocab[0]])
            for w in self.vocab[1:]:
                if w not in emb_arr.keys():
                    emb_arr[w]=np.zeros(vec_size, )
                np.concatenate(self.word_emb_arr, np.array(emb_arr[w]))

        # reading the jsonl for a specific user_idx
        self.read_jsonl(data_jsonl, user_idx, test_only)

        x, _, _ = self.__getitem__(0)  # x, labels, id


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        idx = idx % len(self.utt_list)
        x = self.x[idx]
        y = self.y[idx]

        # Convert words to embeddings
        line = np.array(line_to_indices(x['src_text'], self.indd))
        x = self.word_emb_arr[line]
        x = np.array(x)
        y = np.array(y)

        return x, y, self.utt_list[idx]


    def read_jsonl(self, jsonl_path, user_idx, test_only):
        """
        reads in a jsonl file for a specific user (unless it's for val/testing) and returns a list of embeddings and targets.
        """
        if isinstance(jsonl_path, str):
            assert os.path.exists(jsonl_path) is True, "Missing a JSONL file for Text dataloader: {}".format(jsonl_path)
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
        utt_list=[]
        if test_only:
            for user in self.user_list:
                utt={}
                for e in raw_x_batch[user]['x']:
                    utt['src_text'] = e[4]
                    utt['duration'] = len(utt["src_text"].split(' '))  # to orginize the batches
                    utt_list.append(utt)
        else:
            for e in raw_x_batch['x']:
                utt={}
                utt['src_text'] = e[4]
                utt['duration'] = len(utt["src_text"].split(' '))
                utt_list.append(utt)

        for utt in utt_list:
            utt["loss_weight"] = 1.0
            self.utt_list.append(utt)
        return utt_list


    def process_y(self, raw_y_batch, test_only=False):
        if test_only:
            y_batch = [e for i in self.user_list for e in raw_y_batch[i]['y']]
        else:
            y_batch = raw_y_batch['y']
        y_batch = np.array(y_batch)
        return y_batch

