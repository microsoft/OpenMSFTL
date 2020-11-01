import numpy as np
import random

import torch
from torch.utils.data import DataLoader, sampler
from torch.utils.data.distributed import DistributedSampler

from .image_dataset import ImageDataset
from ftl.data_manager.data_text.data_utils import BatchSampler, DynamicBatchSampler


class ImageDataLoader(DataLoader):
    """
    PyTorch dataloader for loading image data from
    text_dataset.
    """

    def __init__(self, img_size, batch_size, mode, num_workers=0, **kwargs):

        self.img_size = img_size
        # decide on a dataset
        dataset = ImageDataset(
                        data_jsonl = kwargs['data_jsonl'],
                        test_only  = False if mode=="train" else True,
                        user_idx   = kwargs['clientx'])

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
            super(ImageDataLoader, self).__init__(dataset,
                                            batch_size=1,
                                            sampler=sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            drop_last=True,
                                            shuffle=False,
                                            pin_memory=kwargs["pin_memory"])
            return

        if batch_sampler is None:
            super(ImageDataLoader, self).__init__(dataset,
                                            batch_size=batch_size,
                                            sampler=sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            drop_last=True)
        else:
            super(ImageDataLoader, self).__init__(dataset,
                                            batch_sampler=batch_sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.collate_fn,
                                            pin_memory=kwargs["pin_memory"])


    def collate_fn(self, batch):
        srcs, tgts, utt_ids = zip(*batch)
        x_len = [1 for s in srcs]

        return {
                'x': torch.from_numpy(np.array(srcs)).view(-1, self.img_size[0], self.img_size[1]).unsqueeze(1),
                'x_len': x_len,
                'y': torch.from_numpy(np.array(tgts)),
                'y_len': x_len,
                'utt_ids' : utt_ids,
                'total_frames' : sum(x_len),
                'total_frames_with_padding' : 1.0,
                'loss_weight' : None
            }
