# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.communication as comm
from pcode.utils.sparsification import (
    get_n_bits,
    SignCompressor,
    SparsificationCompressor,
    QuantizationCompressor,
)


class ECD_PSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
        model=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ECD_PSGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf

        # define the aggregator.
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        self.aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=self.neighbors_info,
            aggregator_type="decentralized",
        )
        self.world_aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )
        self.init_neighbor_hat_params()

        # related to sparsification/quantization.
        self.compressor = ECDCompressor(
            aggregator=self.aggregator,
            comm_op=conf.comm_op,
            comm_device=conf.comm_device,
            compress_ratio=conf.compress_ratio,
            quantize_level=conf.quantize_level,
            is_biased=conf.is_biased,
            backend=conf.backend,
            use_ipc=conf.use_ipc,
        )

        # define reducer.
        self.backend = conf.backend

    def init_neighbor_hat_params(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)

        # init the neighbor_params.
        self.neighbor_hat_params = dict()
        for rank, _ in self.neighbors_info.items():
            self.neighbor_hat_params[rank] = deepcopy(flatten_params)

    def __setstate__(self, state):
        super(ECD_PSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # Apply the gradients with the weight decay and momentum.
        with kargs["timer"]("grad.apply_grad", epoch=self.conf.epoch_):
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=False
            )

        # get flattened params.
        with kargs["timer"]("grad.get_params", epoch=self.conf.epoch_):
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            flatten_params = TensorBuffer(params)

            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            flatten_grads = TensorBuffer(grads)

        with kargs["timer"]("grad.get_extrapolated_model", epoch=self.conf.epoch_):
            flatten_updated_params = deepcopy(flatten_params)

            # get weighted hat params.
            flatten_updated_params.buffer = sum(
                [
                    _hat_params.buffer * self.neighbors_info[_rank]
                    for _rank, _hat_params in self.neighbor_hat_params.items()
                ]
            )

        # get updated local model (flatten params).
        with kargs["timer"]("grad.unflatten_to_update", epoch=self.conf.epoch_):
            flatten_updated_params.buffer.add_(
                flatten_grads.buffer, alpha=-self.param_groups[0]["lr"]
            )
            flatten_updated_params.unpack(params)

            # get extrapolated model.
            flatten_updated_params.buffer = (
                (1 - 0.5 * self.conf.local_index) * flatten_params.buffer
                + 0.5 * self.conf.local_index * flatten_updated_params.buffer
            )

        # compress the model difference and sync.
        with kargs["timer"]("grad.compress", epoch=self.conf.epoch_):
            sync_buffer = {
                "original_shapes": self.shapes,
                "flatten_updated_params": flatten_updated_params,
            }
            self.compressor.compress(sync_buffer)

        with kargs["timer"]("grad.sync", epoch=self.conf.epoch_):
            self.compressor.sync(sync_buffer)

        with kargs["timer"]("grad.unflatten_to_update", epoch=self.conf.epoch_):
            self.compressor.uncompress(
                sync_buffer, self.neighbor_hat_params, self.conf.local_index
            )
        return sync_buffer["n_bits"]


"""the entry for ECDCompressor."""


class ECDCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "top_k" in kargs["comm_op"] or "random_k" in kargs["comm_op"]:
            self.compressor_fn = ECDSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = ECDQuantizationCompressor(**kargs)
        elif "sign" in kargs["comm_op"]:
            self.compressor_fn = ECDSignCompressor(**kargs)
        else:
            raise NotImplementedError

    def compress(self, *args, **kargs):
        return self.compressor_fn.compress(*args, **kargs)

    def sync(self, *args, **kargs):
        return self.compressor_fn.sync(*args, **kargs)

    def uncompress(self, *args, **kargs):
        return self.compressor_fn.uncompress(*args, **kargs)


"""Detailed ECDCompressors, e.g., top-k/random-k, quantization, sign-based quantization."""


class ECDSparsificationCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        **kargs
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        selected_values, selected_indices = [], []

        for flatten_updated_param in sync_buffer["flatten_updated_params"]:
            _selected_values, _selected_indices = self.compressor_fn.compress(
                flatten_updated_param, self.comm_op, self.compress_ratio, self.is_biased
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

        # get selected shapes.
        selected_shapes = [len(_value) for _value in selected_values]

        # flatten selected values/indices.
        flatten_selected_values = TensorBuffer(selected_values)
        flatten_selected_indices = TensorBuffer(selected_indices)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
            flatten_selected_indices.buffer
        )

        # update shared dict.
        sync_buffer["selected_shapes"] = selected_shapes
        sync_buffer["flatten_selected_values"] = flatten_selected_values
        sync_buffer["flatten_selected_indices"] = flatten_selected_indices
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # get the flatten values.
        message_to_send = torch.cat(
            [
                sync_buffer["flatten_selected_values"].buffer,
                sync_buffer["flatten_selected_indices"].buffer,
            ]
        )

        if self.comm_device == "cpu":
            message_to_send = message_to_send.cpu().pin_memory()

        # sync.
        synced_message = self.aggregator_fn._agg(
            message_to_send, op="get_raw_sync_data", force_wait=True
        )

        # update sync_buffer.
        sync_buffer["synced_message"] = synced_message
        sync_buffer["sycned_message_size"] = len(message_to_send)

    def uncompress(self, sync_buffer, neighbor_hat_params, local_index):
        sycned_message_size = int(sync_buffer["sycned_message_size"] / 2)

        # uncompress and update.
        for rank, hat_params in neighbor_hat_params.items():
            _message = comm.recover_device(
                sync_buffer["synced_message"][rank], device=hat_params.buffer.device
            )
            values = _message[:sycned_message_size]
            indices = _message[sycned_message_size:]

            # deal with unbalanced values/indieces
            q_values, q_indices = self.compressor_fn.uncompress(
                values,
                indices,
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # update the flatten hat params.
            hat_params.buffer[q_indices] = (
                hat_params.buffer[q_indices]
                .mul(1 - 2 / local_index)
                .add(2 / local_index, q_values)
            )


class ECDQuantizationCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        **kargs
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.kargs = kargs
        self.compressor_fn = QuantizationCompressor()

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        quantized_values = []

        for flatten_updated_param in sync_buffer["flatten_updated_params"]:
            _quantized_values = self.compressor_fn.compress(
                flatten_updated_param, self.comm_op, self.quantize_level, self.is_biased
            )
            quantized_values.append(_quantized_values)

        # flatten selected values/indices.
        flatten_updates = TensorBuffer(quantized_values)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_updates.buffer) * self.quantize_level / 32

        # update shared dict.
        sync_buffer["flatten_updates"] = flatten_updates
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare the sync.
        to_sync_message = sync_buffer["flatten_updates"].buffer

        if self.comm_device == "cpu":
            to_sync_message = to_sync_message.cpu().pin_memory()

        # sync.
        synced_message = self.aggregator_fn._agg(
            to_sync_message, op="get_raw_sync_data", force_wait=True
        )

        # update sync_buffer.
        sync_buffer["synced_message"] = synced_message

    def uncompress(self, sync_buffer, neighbor_hat_params, local_index):
        # uncompress and update.
        for rank, hat_params in neighbor_hat_params.items():
            # map the tensors to the correct location.
            _message = comm.recover_device(
                sync_buffer["synced_message"][rank], device=hat_params.buffer.device
            )

            # update the flatten hat params.
            hat_params.buffer.mul_(1 - 2 / local_index).add_(
                _message, alpha=2 / local_index
            )


class ECDSignCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        **kargs
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.kargs = kargs
        self.compressor_fn = SignCompressor()

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        norms, updates = [], []
        for flatten_updated_param in sync_buffer["flatten_updated_params"]:
            _update = flatten_updated_param
            updates += [_update]
            norms += [_update.norm(p=1)]

        # flatten selected values/indices.
        flatten_norms = TensorBuffer(norms)
        flatten_updates = TensorBuffer(updates)
        signs, sign_size = self.compressor_fn.compress(flatten_updates.buffer)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_norms.buffer) + get_n_bits(signs)

        # update shared dict.
        sync_buffer["flatten_norms"] = flatten_norms
        sync_buffer["flatten_updates"] = flatten_updates
        sync_buffer["signs"] = signs
        sync_buffer["sign_size"] = sign_size
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare sync.
        to_sync_flatten_norms = sync_buffer["flatten_norms"].buffer
        to_sync_signs = sync_buffer["signs"]

        if self.comm_device == "cpu":
            to_sync_flatten_norms = to_sync_flatten_norms.cpu().pin_memory()
            to_sync_signs = to_sync_signs.cpu().pin_memory()

        # sync.
        synced_flatten_norms = self.aggregator_fn._agg(
            sync_buffer["flatten_norms"].buffer, op="get_raw_sync_data", force_wait=True
        )
        synced_signs = self.aggregator_fn._agg(
            sync_buffer["signs"], op="get_raw_sync_data", force_wait=True
        )

        # update sync_buffer.
        sync_buffer["synced_flatten_norms"] = synced_flatten_norms
        sync_buffer["synced_signs"] = synced_signs

    def uncompress(self, sync_buffer, neighbor_hat_params, local_index):
        # uncompress and update.
        for rank, hat_params in neighbor_hat_params.items():
            # recover the message and the corresponding device.
            sync_buffer["flatten_norms"].buffer = comm.recover_device(
                sync_buffer["synced_flatten_norms"][rank],
                device=hat_params.buffer.device,
            )
            sync_buffer["flatten_updates"].buffer = self.compressor_fn.uncompress(
                comm.recover_device(
                    sync_buffer["synced_signs"][rank], device=hat_params.buffer.device
                ),
                sync_buffer["sign_size"],
            )

            # update hat_params.
            for hat_param, norm, sign in zip(
                hat_params, sync_buffer["flatten_norms"], sync_buffer["flatten_updates"]
            ):
                # update the flatten hat params.
                hat_param.mul_(1 - 2 / local_index).add_(
                    2 / local_index * norm / sign.nelement(), sign
                )
