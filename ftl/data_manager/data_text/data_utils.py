import numpy as np
import operator
import torch
import wave
import json
import random
import scipy.signal

from torch.utils.data import Dataset, DataLoader, sampler
import torch.autograd as autograd

# PART of dirty hack
from torch.utils.data.distributed import DistributedSampler


# instead of inferiting from sampler.Sampler, can you inherit from sampler.DistributedSamppler?
# Not that simple. A very quick, dirty and inefficient hack is to use DistributedSampler for now
# directly in the SRDataLoader
class BatchSampler(sampler.Sampler):
    """
    Simply determines the order in which the loader will read samples from the data set.
    We want to sample batches randomly, but each batch should have samples that are
    close to each other in the dataset (so that we don't have a lot of zero padding)
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        batches = [range(begin_id, begin_id + batch_size) for begin_id in range(0, len(dataset), batch_size)]

        # if the indexes in the last batch are going over len(dataset), we drop the last batch.
        # Not entirely happy with this. But should serve all purposes for validation and test
        if batches[-1][-1] >= len(dataset):
            del batches[-1]

        self.batches = batches

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(idx for batch in self.batches for idx in batch)

    def __len__(self):
        return len(self.batches) * self.batch_size


class DynamicBatchSampler(sampler.Sampler):
    """Extension of Sampler that will do the following:
        1.  Change the batch size (essentially number of sequences)
            in a batch to ensure that the total number of frames are less
            than a certain threshold.
        2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(self, sampler, frames_threshold, max_batch_size=0, unsorted_batch=False, fps= 1000 / 30):
        """
        @sampler: will mostly be an instance of DistributedSampler.
        Though it should work with any sampler.
        @frames_threshold: maximum area of the batch
        """
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_batch_size = max_batch_size
        self.unsorted_batch = unsorted_batch

        indices, batches = list(), list()
        # the dataset to which these indices are pointing to
        dataset = self.sampler.dataset
        # get all the indices and corresponding durations from
        # the sampler
        for idx in self.sampler:
            indices.append((idx, dataset.utt_list[idx]["duration"]))
        # sort the indices according to duration
        if self.unsorted_batch is False:
            indices.sort(key=lambda elem : elem[1])
            max_dur = indices[-1][1]
        else:
            # make sure that you will be able to serve all the utterances
            max_dur = max([indices[i][1] for i in range(len(indices))])

        assert max_dur < self.frames_threshold, ("Won't be able"
                "to serve all sequences. frames_threshold={} while longest"
                " utterance has {} frames").format(self.frames_threshold,
                                                    max_dur)

        # start clubbing the utterances together
        batch = list()
        batch_frames, batch_area = 0, 0
        max_frames_in_batch = 0
        for idx, duration in indices:
            if duration > 0:
                frames = duration * fps
                if frames > max_frames_in_batch:
                    max_frames_in_batch = frames

                # consider adding this utterance to the current batch
                if (self.unsorted_batch and (len(batch)+1)*max_frames_in_batch <= self.frames_threshold and (max_batch_size == 0 or len(batch) < max_batch_size)) \
                    or (not self.unsorted_batch and batch_frames + frames <= self.frames_threshold and (max_batch_size == 0 or len(batch) < max_batch_size)): # this line is to keep the behavior of old code base without "unsorted_batch"
                    # can add to the batch
                    batch.append(idx)
                    batch_frames += frames
                    batch_area = max_frames_in_batch * len(batch)
                else:
                    # log the stats and add previous batch to batches
                    if batch_area > 0 and len(batch) > 0:
                        batches.append(batch)
                    # make a new one
                    batch = list([idx])
                    batch_frames, batch_area = frames, frames
                    max_frames_in_batch = batch_frames
        if batch_area > 0 and len(batch) > 0:
            batches.append(batch)

        # don't need the 'indices' any more
        del indices
        self.batches = batches

    def __iter__(self):
        # shuffle on a batch level
        random.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

