# -*- coding: utf-8 -*-
import random

import torch
import torch.distributed as dist


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, conf, data, partition_sizes, partition_type="random"):
        # prepare info.
        self.conf = conf
        self.data = data
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.data_size = len(self.data)
        self.partitions = []

        # get unshuffled indices.
        indices = [x for x in range(0, self.data_size)]

        # apply partition function.
        self.partition_indices(indices)

    def partition_indices(self, indices):
        indices = self._get_consistent_indices(indices)

        # partition indices.
        from_index = 0
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index

    def _get_consistent_indices(self, indices):
        if self.conf.graph.rank == 0:
            if self.partition_type == "random":
                # it will randomly shuffle the indices.
                random.shuffle(indices)
            elif self.partition_type == "sorted":
                # it will sort the indices based on the data label.
                indices = [
                    i[0]
                    for i in sorted(enumerate(self.data.targets), key=lambda x: x[1])
                ]

        # sync the indices over nodes.
        indices = torch.IntTensor(indices)
        indices = indices.cuda() if self.conf.backend == "nccl" else indices
        group = dist.new_group(self.conf.graph.ranks)
        dist.broadcast(indices, src=0, group=group)
        indices = indices.cpu() if self.conf.backend == "nccl" else indices
        return list(indices)

    def use(self, partition_ind):
        return Partition(self.data, self.partitions[partition_ind])
