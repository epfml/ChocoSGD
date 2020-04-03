# -*- coding: utf-8 -*-
import math

import torch
from torch.optim.optimizer import Optimizer, required

from pcode.utils.communication import (
    get_aggregators,
    get_data,
    flatten,
    unflatten,
    recover_device,
)
from pcode.utils.sparsification import (
    QuantizationCompressor,
    SparsificationCompressor,
    get_n_bits,
)
import pcode.utils.communication as comm


class DGC(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
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
        super(DGC, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.n_nodes = conf.graph.n_nodes
        self.rank = conf.graph.rank

        # define the aggregator.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )
        self.world_aggregator = get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        # related to sparsification/quantization.
        self.comm_op = conf.comm_op
        self.comm_device = conf.comm_device
        self.is_compress_op = "compress" in self.comm_op
        self.compress_ratio = conf.compress_ratio
        self.compress_warmup_values = conf.compress_warmup_values
        self.compress_warmup_epochs = conf.compress_warmup_epochs
        self.quantize_level = conf.quantize_level
        self.is_biased = conf.is_biased

        self.clip_grad = conf.clip_grad
        self.clip_grad_val = conf.clip_grad_val
        self.mask_momentum = conf.mask_momentum

        self.init_memory()
        self.init_compression()

        # define compressors.
        if self.is_compress_op:
            self.compressor_fn = SparsificationCompressor()
        else:
            self.compressor_fn = QuantizationCompressor()

        # define reducer.
        self.backend = conf.backend

    def init_memory(self):
        self.memory_of_grads = dict()

        for group in self.param_groups:
            for p in group["params"]:
                self.memory_of_grads[group["name"]] = torch.zeros_like(p.data).view(-1)

    def init_compression(self):
        # configure gradient warmup values
        if self.compress_ratio is not None:
            compress_warmup_values = [
                float(value) for value in self.compress_warmup_values.split(",")
            ]
            self.compress_warmup_values = [
                value
                for value in compress_warmup_values
                if value <= self.compress_ratio
            ]

            num_compress_warmup_values = len(self.compress_warmup_values)
            self.detailed_compress_warmup_epochs = [
                1.0 * ind / num_compress_warmup_values * self.compress_warmup_epochs
                for ind in range(1, num_compress_warmup_values + 1)
            ]

    def __setstate__(self, state):
        super(DGC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # apply local gradient.
        with kargs["timer"]("grad.apply_grad", epoch=self.conf.epoch_):
            self._apply_gradient()

        # Unflatten the saved hat params.
        with kargs["timer"]("grad.recover_hat_params", epoch=self.conf.epoch_):
            params, _ = get_data(self.param_groups, self.param_names, is_get_grad=False)
            grads, shapes = get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )

        # compress.
        with kargs["timer"]("grad.compress", epoch=self.conf.epoch_):
            selected_values, selected_indices, n_bits = self._compress(grads)

        # sync.
        with kargs["timer"]("grad.sync", epoch=self.conf.epoch_):
            synced_message, message_size = self._sync(selected_values, selected_indices)

        # recover and update the neighbor hat params.
        with kargs["timer"]("grad.recover_info", epoch=self.conf.epoch_):
            updated_flatten_params = self._recover_info(
                flatten(params),
                synced_message,
                message_size,
                self.selected_shapes,
                shapes,
            )

        with kargs["timer"]("grad.update_model", epoch=self.conf.epoch_):
            # finally unflatten.
            unflatten(params, updated_flatten_params, shapes)
        return n_bits

    def _compress(self, grads):
        selected_values, selected_indices, n_bits = [], [], []
        for (idx, param_name), grad in zip(self.param_names, grads):
            # add memory back.
            _grad = grad.data.view(-1) + self.memory_of_grads[param_name]

            # get values and indices
            compress_ratio = self._get_compress_ratio()
            _selected_values, _selected_indices, _n_bits = compress_or_quantize(
                grad=_grad,
                comm_op=self.comm_op,
                compressor_fn=self.compressor_fn,
                compress_ratio=compress_ratio,
                quantize_level=self.quantize_level,
                is_biased=self.is_biased,
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)
            n_bits.append(_n_bits)

            # update the memory
            if self.is_compress_op:
                _, nmask = self.compressor_fn.get_mask(_grad, _selected_indices)
                self.memory_of_grads[param_name] = _grad * nmask

                # apply momentum factor masking.
                if self.mask_momentum:
                    self.state[self.param_groups[idx]["params"][0]][
                        "momentum_buffer"
                    ].mul_(nmask.view(grad.size()))
            else:
                # self.memory_of_grads[param_name] = _grad - _selected_values
                pass

        # get selected shapes.
        self.selected_shapes = [len(_value) for _value in selected_values]

        # flatten selected values/indices.
        flatten_selected_values = flatten(selected_values)
        flatten_selected_indices = (
            flatten(selected_indices) if selected_indices[0] is not None else None
        )
        return flatten_selected_values, flatten_selected_indices, sum(n_bits)

    def _sync(self, selected_values, selected_indices):
        if self.is_compress_op:
            # concate values and indices.
            message_to_send = torch.cat([selected_values, selected_indices])

            if self.comm_device == "cpu":
                message_to_send = message_to_send.cpu().pin_memory()

            synced_message = self.world_aggregator._agg(
                message_to_send, communication_scheme="all_gather"
            )
        else:
            message_to_send = selected_values

            if self.comm_device == "cpu":
                message_to_send = message_to_send.cpu().pin_memory()

            synced_message = self.world_aggregator._agg(
                message_to_send, op="sum", communication_scheme="all_reduce"
            )

        # get message size.
        message_size = len(message_to_send)
        return synced_message, message_size

    def _recover_info(
        self, flatten_params, synced_message, message_size, selected_shapes, shapes
    ):
        # use the pointers to recover the info and get synced grad.
        _message_size = int(message_size / 2)

        if self.is_compress_op:
            empty_grads = torch.zeros_like(flatten_params)

            for message in synced_message:
                q_values, q_indices = self.compressor_fn.uncompress(
                    message[:_message_size],
                    message[_message_size:],
                    selected_shapes,
                    shapes,
                )

                empty_grads[q_indices] += q_values

            # get update tensor.
            _update = empty_grads / self.n_nodes
        else:
            # get update tensor.
            _update = synced_message / self.n_nodes

        # update flatten_params (assume the used lr is the same over params)
        updated_flatten_params = flatten_params.add(
            -self.param_groups[0]["lr"],
            recover_device(_update, device=flatten_params.device),
        )
        return updated_flatten_params

    def _clip_gradient(self, grad, param_state, scale=True):
        # calculate the grad norm.
        grad_norm = grad.norm(p=2)

        threshold = self.clip_grad_val
        if threshold is None:
            return grad
        else:
            threshold *= 1.0 if not scale else 1.0 / math.sqrt(self.n_nodes)
            if grad_norm >= threshold:
                grad = threshold / grad_norm * grad
            return grad

    def _get_compress_ratio(self):
        # if we are under the phase of warmup, use different dgc ratio,
        # otherwise return the expected one.
        if self.is_compress_op:
            if self.conf.epoch_ < self.compress_warmup_epochs:
                for ind, val in enumerate(self.detailed_compress_warmup_epochs):
                    if self.conf.epoch_ < val:
                        return self.compress_warmup_values[ind]
            return self.compress_ratio
        else:
            return None

    def _apply_gradient(self):
        """Performs a single optimization step.

        Avoid to use momentum to accumulate the gradients from other workers.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        for group in self.param_groups:
            # retrieve para.
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                # get param_state
                param_state = self.state[p]

                # get the gradient
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # add the weight decay.
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                # clip the gradient.
                if self.clip_grad:
                    d_p = self._clip_gradient(d_p, param_state)

                # apply the momentum.
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(d_p)
                        buf.add_(d_p)
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.grad.data = d_p


def compress_or_quantize(
    grad, comm_op, compressor_fn, compress_ratio, quantize_level, is_biased
):
    if "compress" in comm_op:
        values, indices = compressor_fn.compress(
            grad, comm_op, compress_ratio, is_biased
        )

        n_bits = get_n_bits(values) + get_n_bits(indices)
    elif "quantize" in comm_op:
        values = compressor_fn.compress(grad, comm_op, quantize_level, is_biased)
        indices = None

        n_bits = get_n_bits(values) * quantize_level / 32
    else:
        raise NotImplementedError
    return values, indices, n_bits
