import numpy as np
import random

import torch
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler

from .text_dataset import TextDataset
from .data_utils import BatchSampler, DynamicBatchSampler


class TextDataLoader(DataLoader):
    """
    PyTorch dataloader for loading text data from
    text_dataset.
    """

    def __init__(self, vec_size, batch_size, mode, num_workers=0, **kwargs):

        self.vec_size=vec_size
        # decide on a dataset
        dataset = TextDataset(
                        data_jsonl   = kwargs['data_jsonl'],
                        max_num_words= batch_size,
                        test_only    = False if mode=="train" else True,
                        vocab_dict   = kwargs['vocab_dict'],
                        user_idx     = kwargs['clientx'])

        # decide on a sampler
        batch_sampler = None
        if mode == 'train':
            sampler = DistributedSampler(dataset,
                                        num_replicas=1,
                                        rank=0)
            sampler.set_epoch(random.randint(0, 10**10))  # set epoch to a random number so that training data is shuffled differently for each dataloader

            batch_sampler = DynamicBatchSampler(sampler,
                                    frames_threshold = batch_size,
                                    max_batch_size   = kwargs['max_batch_size'],
                                    unsorted_batch   = kwargs['unsorted_batch'],
                                    fps=1)

        elif mode == 'val' or mode == 'test':
            sampler = BatchSampler(dataset, batch_size=1)
            super(TextDataLoader, self).__init__(dataset,
                                            batch_size=1,
                                            sampler=sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            drop_last=True,
                                            shuffle=False,
                                            pin_memory=kwargs["pin_memory"])
            return

        if batch_sampler is None:
            super(TextDataLoader, self).__init__(dataset,
                                            batch_size=batch_size,
                                            sampler=sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            drop_last=True)
        else:
            super(TextDataLoader, self).__init__(dataset,
                                            batch_sampler=batch_sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            pin_memory=kwargs["pin_memory"])


    def create_loader(self):
        # TODO: This is a temporal solution to maintain the compatibility
        return self


    def collate_fn(self, batch):
        def pad_and_concat_feats(labels):
            batch_size = len(labels)
            max_len = max(len(l) for l in labels)
            cat_labels = np.zeros((batch_size, max_len, self.vec_size))

            for e, l in enumerate(labels):
                cat_labels[e,:,:] = l
            return cat_labels


        src_seq, tgt_seq, utt_ids = zip(*batch)
        x_len = [len(s) for s in src_seq]

        packed = {
                    'x': torch.from_numpy(pad_and_concat_feats(src_seq)),
                    'x_len': [len(s) for s in src_seq],
                    'y': torch.from_numpy(np.array(tgt_seq)),
                    'y_len': [len(tgt_seq)],
                    'utt_ids' : utt_ids,
                    'total_frames' : sum(x_len),
                    'total_frames_with_padding' : 1.0,
                    'loss_weight' : None
                }

        return packed
