# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
from pcode.utils.sparsification import (
    get_n_bits,
    SignCompressor,
    SparsificationCompressor,
    QuantizationCompressor,
)
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.communication as comm


class DeepSqueeze(Optimizer):
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
        super(DeepSqueeze, self).__init__(params, defaults)

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
        self.init_memory()

        # related to sparsification/quantization.
        self.compressor = DeepSqueezeCompressor(
            aggregator=self.aggregator,
            rank=self.rank,
            comm_op=conf.comm_op,
            comm_device=conf.comm_device,
            compress_ratio=conf.compress_ratio,
            quantize_level=conf.quantize_level,
            is_biased=conf.is_biased,
            backend=conf.backend,
            use_ipc=conf.use_ipc,
            consensus_stepsize=conf.consensus_stepsize,
        )

    def init_memory(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.memory = TensorBuffer(params)
        self.memory.buffer = torch.zeros_like(self.memory.buffer)

    def __setstate__(self, state):
        super(DeepSqueeze, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # Apply the gradients with the weight decay and momentum.
        with kargs["timer"]("grad.apply_grad", epoch=self.conf.epoch_):
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=True
            )

        with kargs["timer"]("grad.get_params", epoch=self.conf.epoch_):
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            params_tb = TensorBuffer(params)

        with kargs["timer"]("grad.error_compensate", epoch=self.conf.epoch_):
            self.memory.buffer += params_tb.buffer

        with kargs["timer"]("grad.compress", epoch=self.conf.epoch_):
            sync_buffer = {"original_shapes": self.shapes, "params_tb": self.memory}
            local_compressed_params_tb = self.compressor.compress(sync_buffer)

        with kargs["timer"]("grad.update_memory", epoch=self.conf.epoch_):
            self.memory.buffer = self.memory.buffer - local_compressed_params_tb.buffer

        with kargs["timer"]("grad.sync", epoch=self.conf.epoch_):
            self.compressor.sync(sync_buffer)

        # update local model.
        with kargs["timer"]("grad.decompress", epoch=self.conf.epoch_):
            aggregated_info_tb = self.compressor.uncompress(
                sync_buffer, self.neighbors_info
            )
            params_tb.buffer += aggregated_info_tb.buffer
            params_tb.unpack(params)
        return sync_buffer["n_bits"]


"""the entry for DCDCompressor."""


class DeepSqueezeCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "top_k" in kargs["comm_op"] or "random_k" in kargs["comm_op"]:
            self.compressor_fn = DeepSqueezeSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = DeepSqueezeQuantizationCompressor(**kargs)
        elif "sign" in kargs["comm_op"]:
            self.compressor_fn = DeepSqueezeSignCompressor(**kargs)
        else:
            raise NotImplementedError

    def compress(self, *args, **kargs):
        return self.compressor_fn.compress(*args, **kargs)

    def sync(self, *args, **kargs):
        return self.compressor_fn.sync(*args, **kargs)

    def uncompress(self, *args, **kargs):
        return self.compressor_fn.uncompress(*args, **kargs)


"""Detailed DCDCompressors, e.g., top-k/random-k, quantization, sign-based quantization."""


class DeepSqueezeSparsificationCompressor(object):
    def __init__(
        self,
        aggregator,
        rank,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        consensus_stepsize,
        **kargs
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.rank = rank
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.consensus_stepsize = consensus_stepsize
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        selected_values, selected_indices = [], []

        # compress and get compressed model.
        local_compressed_params_tb = deepcopy(sync_buffer["params_tb"])
        local_compressed_params_tb.buffer = torch.zeros_like(
            local_compressed_params_tb.buffer
        )
        for param, local_compressed_param in zip(
            sync_buffer["params_tb"], local_compressed_params_tb
        ):
            _selected_values, _selected_indices = self.compressor_fn.compress(
                param, self.comm_op, self.compress_ratio, self.is_biased
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

            # update the local compressed params.
            local_compressed_param.data = local_compressed_param.data.view(-1)
            local_compressed_param.data[_selected_indices] = _selected_values
            local_compressed_param.data.view(*param.size())

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
        return local_compressed_params_tb

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

    def uncompress(self, sync_buffer, neighbors_info):
        aggregated_info_tb = deepcopy(sync_buffer["params_tb"])
        aggregated_info_tb.buffer = torch.zeros_like(aggregated_info_tb.buffer)

        # uncompress and update.
        sycned_message_size = int(sync_buffer["sycned_message_size"] / 2)

        for rank in neighbors_info.keys():
            _message = comm.recover_device(
                sync_buffer["synced_message"][rank],
                device=sync_buffer["params_tb"].buffer.device,
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
            aggregated_info_tb.buffer[q_indices] += (
                self.consensus_stepsize
                * (neighbors_info[rank] - (1 if rank == self.rank else 0))
                * q_values
            )
        return aggregated_info_tb


class DeepSqueezeQuantizationCompressor(object):
    def __init__(
        self,
        aggregator,
        rank,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        consensus_stepsize,
        **kargs
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.rank = rank
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.consensus_stepsize = consensus_stepsize
        self.kargs = kargs
        self.compressor_fn = QuantizationCompressor()

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        quantized_values = []

        # compress and get compressed model.
        local_compressed_params_tb = deepcopy(sync_buffer["params_tb"])
        local_compressed_params_tb.buffer = torch.zeros_like(
            local_compressed_params_tb.buffer
        )
        for param, local_compressed_param in zip(
            sync_buffer["params_tb"], local_compressed_params_tb
        ):
            # quantize.
            _quantized_values = self.compressor_fn.compress(
                param, self.comm_op, self.quantize_level, self.is_biased
            )
            quantized_values.append(_quantized_values)

            # update the local compressed params.
            local_compressed_param.data.copy_(_quantized_values)

        # flatten selected values/indices.
        flatten_updates = TensorBuffer(quantized_values)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_updates.buffer) * self.quantize_level / 32

        # update shared dict.
        sync_buffer["flatten_updates"] = flatten_updates
        sync_buffer["n_bits"] = n_bits
        return local_compressed_params_tb

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

    def uncompress(self, sync_buffer, neighbors_info):
        aggregated_info_tb = deepcopy(sync_buffer["params_tb"])
        aggregated_info_tb.buffer = torch.zeros_like(aggregated_info_tb.buffer)

        # uncompress and update.
        for rank in neighbors_info.keys():
            # map the tensors to the correct location.
            _message = comm.recover_device(
                sync_buffer["synced_message"][rank],
                device=sync_buffer["params_tb"].buffer.device,
            )

            # update the flatten hat params.
            aggregated_info_tb.buffer.add_(
                self.consensus_stepsize
                * (neighbors_info[rank] - (1 if rank == self.rank else 0))
                * _message
            )
        return aggregated_info_tb


class DeepSqueezeSignCompressor(object):
    def __init__(
        self,
        aggregator,
        rank,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        consensus_stepsize,
        **kargs
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.rank = rank
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.consensus_stepsize = consensus_stepsize
        self.kargs = kargs
        self.compressor_fn = SignCompressor()

    def compress(self, sync_buffer):
        # flatten selected values/indices.
        param_norms_tb = TensorBuffer(
            [param.norm(p=1) for param in sync_buffer["params_tb"]]
        )
        signs, sign_size = self.compressor_fn.compress(sync_buffer["params_tb"].buffer)

        # get compressed model.
        local_compressed_params_tb = deepcopy(sync_buffer["params_tb"])
        for local_compressed_param, param_norm, param in zip(
            local_compressed_params_tb, param_norms_tb, sync_buffer["params_tb"]
        ):
            local_compressed_param.data.copy_(
                param_norm * torch.sign(param) / param.nelement()
            )

        # get n_bits to transmit.
        n_bits = get_n_bits(param_norms_tb.buffer) + get_n_bits(signs)

        # update shared dict.
        sync_buffer["param_norms_tb"] = param_norms_tb
        sync_buffer["signs"] = signs
        sync_buffer["sign_size"] = sign_size
        sync_buffer["n_bits"] = n_bits
        return local_compressed_params_tb

    def sync(self, sync_buffer):
        # prepare sync.
        to_sync_param_norms = sync_buffer["param_norms_tb"].buffer
        to_sync_signs = sync_buffer["signs"]

        if self.comm_device == "cpu":
            to_sync_param_norms = to_sync_param_norms.cpu().pin_memory()
            to_sync_signs = to_sync_signs.cpu().pin_memory()

        # sync.
        synced_param_norms = self.aggregator_fn._agg(
            to_sync_param_norms, op="get_raw_sync_data", force_wait=True
        )
        synced_signs = self.aggregator_fn._agg(
            to_sync_signs, op="get_raw_sync_data", force_wait=True
        )

        # update sync_buffer.
        sync_buffer["synced_param_norms"] = synced_param_norms
        sync_buffer["synced_signs"] = synced_signs

    def uncompress(self, sync_buffer, neighbors_info):
        aggregated_info_tb = deepcopy(sync_buffer["params_tb"])
        aggregated_info_tb.buffer = torch.zeros_like(aggregated_info_tb.buffer)

        # uncompress and update.
        for rank in neighbors_info.keys():
            param_norms = sync_buffer["synced_param_norms"][rank]
            signs = sync_buffer["synced_signs"][rank]

            # recover the message and the corresponding device.
            param_norms = comm.recover_device(
                param_norms, device=sync_buffer["params_tb"].buffer.device
            )
            signs = self.compressor_fn.uncompress(
                comm.recover_device(
                    signs, device=sync_buffer["params_tb"].buffer.device
                ),
                sync_buffer["sign_size"],
            )

            # build the corresponding tensorbuffer.
            param_norms_tb = TensorBuffer(param_norms)
            signs_tb = deepcopy(sync_buffer["params_tb"])
            signs_tb.buffer = signs

            # accumulate information for the neighborhood..
            for _info, _param_norm, _sign in zip(
                aggregated_info_tb, param_norms_tb, signs_tb
            ):
                _info.add_(
                    self.consensus_stepsize
                    * (neighbors_info[rank] - (1 if rank == self.rank else 0))
                    * (_param_norm / _sign.nelement() * _sign)
                )
        return aggregated_info_tb
