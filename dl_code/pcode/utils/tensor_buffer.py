# -*- coding: utf-8 -*-
from copy import deepcopy
from pcode.utils.communication import flatten


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors, use_cuda=True):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = flatten(tensors, use_cuda=use_cuda)  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index]: self._end_idx[index]].view(
            self._tensors[index].size())

    def __len__(self):
        return len(self._tensors)

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor.data[:] = entry

    def _mathop(self, resulted_buffer):
        _tensors = deepcopy(self._tensors)

        for tensor, entry in zip(_tensors, resulted_buffer):
            tensor.data[:] = entry
        return TensorBuffer(_tensors)

    def __add__(self, other):
        assert isinstance(other, TensorBuffer)
        return self._mathop(self.buffer + other.buffer)

    def __sub__(self, other):
        assert isinstance(other, TensorBuffer)
        return self._mathop(self.buffer - other.buffer)

    def __mul__(self, other):
        if isinstance(other, TensorBuffer):
            return self._mathop(self.buffer * other.buffer)
        else:
            return self._mathop(self.buffer * other)
