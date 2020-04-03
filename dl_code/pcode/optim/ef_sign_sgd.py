# -*- coding: utf-8 -*-
import copy
import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits, SignCompressor
from pcode.utils.tensor_buffer import TensorBuffer


class EF_SignSGD(Optimizer):
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
        super(EF_SignSGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()

        # define the aggregator.
        self.world_aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # for EF-signSGD.
        self.init_memory()
        self.compressor = EFSignCompressor(
            rank=self.rank,
            world_size=len(conf.graph.ranks),
            aggregator=self.world_aggregator,
            comm_op=conf.comm_op,
            comm_device=self.conf.comm_device,
            use_ipc=conf.use_ipc,
        )

    def init_memory(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        self.memory_tb = TensorBuffer(params)
        self.memory_tb.buffer = torch.zeros_like(self.memory_tb.buffer)

    def __setstate__(self, state):
        super(EF_SignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=False
            )

        with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
            # Get data.
            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            grads_tb = TensorBuffer(grads)

        with kargs["timer"]("sync.use_memory", epoch=self.conf.epoch_):
            # use memory.
            grads_tb.buffer.add_(self.memory_tb.buffer)

        with kargs["timer"]("sync.compress", epoch=self.conf.epoch_):
            # compress.
            sync_buffer = self.compressor.compress(grads_tb)

        with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
            self.compressor.sync(sync_buffer)

        with kargs["timer"]("sync.update_memory", epoch=self.conf.epoch_):
            # update memory.
            self.memory_tb.buffer = (
                grads_tb.buffer - sync_buffer["synced_grads_tb"].buffer
            )

        with kargs["timer"]("sync.decompress", epoch=self.conf.epoch_):
            sync_grads_tb = self.compressor.decompress(sync_buffer)

        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            # appply the gradient but only with the gradient.
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            params_tb = TensorBuffer(params)

            # apply the gradient.
            params_tb.buffer.add_(-self.param_groups[0]["lr"] * sync_grads_tb.buffer)

            # unpack.
            params_tb.unpack(params)
        return sync_buffer["n_bits"]


class EFSignCompressor(object):
    def __init__(
        self, rank, world_size, aggregator, comm_op, comm_device, use_ipc, **kargs
    ):
        # assign the common hyper-parameters
        self.rank = rank
        self.world_size = world_size
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.use_ipc = use_ipc
        self.kargs = kargs
        self.compressor_fn = SignCompressor()

    def compress(self, grads_tb):
        # get the sign/magnitude for the tensor (to be transmitted).
        sync_buffer = dict()

        # flatten selected values/indices.
        grad_norms_tb = TensorBuffer([grad.norm(p=1) for grad in grads_tb])
        signs, sign_size = self.compressor_fn.compress(grads_tb.buffer)

        # get compressed grad.
        synced_grads_tb = copy.deepcopy(grads_tb)
        for synced_grad, grad_norm, grad in zip(
            synced_grads_tb, grad_norms_tb, grads_tb
        ):
            synced_grad.data.copy_(grad_norm * torch.sign(grad) / grad.nelement())

        # get n_bits to transmit.
        n_bits = get_n_bits(grad_norms_tb.buffer) + get_n_bits(signs)

        # update shared dict.
        sync_buffer["grad_norms_tb"] = grad_norms_tb
        sync_buffer["grads_tb"] = grads_tb
        sync_buffer["synced_grads_tb"] = synced_grads_tb
        sync_buffer["signs"] = signs
        sync_buffer["sign_size"] = sign_size
        sync_buffer["n_bits"] = n_bits
        return sync_buffer

    def sync(self, sync_buffer):
        # prepare sync.
        to_sync_grad_norms = sync_buffer["grad_norms_tb"].buffer
        to_sync_signs = sync_buffer["signs"]

        if self.comm_device == "cpu":
            to_sync_grad_norms = to_sync_grad_norms.cpu().pin_memory()
            to_sync_signs = to_sync_signs.cpu().pin_memory()

        # sync.
        synced_grad_norms = self.aggregator_fn._agg(
            to_sync_grad_norms, communication_scheme="all_gather", async_op=False
        )
        synced_signs = self.aggregator_fn._agg(
            to_sync_signs, communication_scheme="all_gather", async_op=False
        )

        # update sync_buffer.
        sync_buffer["synced_grad_norms"] = synced_grad_norms
        sync_buffer["synced_signs"] = synced_signs

    def decompress(self, sync_buffer):
        # decompress and update.
        for rank in range(self.world_size):
            if rank == self.rank:
                continue

            # get grad_norm and build its tensorbuffer.
            _grad_norms = comm.recover_device(
                sync_buffer["synced_grad_norms"][rank],
                device=sync_buffer["synced_grads_tb"].buffer.device,
            )
            grad_norms_tb = TensorBuffer(_grad_norms)

            # get signs and build its tensorbuffer.
            signs = comm.recover_device(
                sync_buffer["synced_signs"][rank],
                device=sync_buffer["synced_grads_tb"].buffer.device,
            )
            _signs = self.compressor_fn.uncompress(signs, sync_buffer["sign_size"])
            signs_tb = copy.deepcopy(sync_buffer["synced_grads_tb"])
            signs_tb.buffer = _signs

            # update grads.
            for grad_norm, sign, synced_grad in zip(
                grad_norms_tb, signs_tb, sync_buffer["synced_grads_tb"]
            ):
                _update = grad_norm * sign / synced_grad.nelement()
                synced_grad.add_(_update)

        # average grad.
        sync_buffer["synced_grads_tb"].buffer /= self.world_size * 1.0
        return sync_buffer["synced_grads_tb"]
