# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits, SignCompressor
from pcode.utils.tensor_buffer import TensorBuffer


class SignSGD(Optimizer):
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
        super(SignSGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        self.local_step = conf.local_step
        self.turn_on_local_step_from_epoch = conf.turn_on_local_step_from

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

        # initialize the concensus
        self.compressor = ExactSignCompressor(
            rank=self.rank,
            world_size=len(conf.graph.ranks),
            majority_vote=conf.majority_vote,
            aggregator=self.world_aggregator,
            comm_op=conf.comm_op,
            comm_device=self.conf.comm_device,
            use_ipc=conf.use_ipc,
        )

    def __setstate__(self, state):
        super(SignSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        # do the local update steps.
        with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
            # get parmas.
            params, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=False
            )
            params_tb = TensorBuffer(params)

        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            # prepare the gradient (sign)
            utils.apply_gradient(
                self.param_groups, self.state, apply_grad_to_model=False
            )
            # get grads.
            grads, _ = comm.get_data(
                self.param_groups, self.param_names, is_get_grad=True
            )
            grads_tb = TensorBuffer(grads)

        # enter the global sync if it satisfies the condition.
        # get the params difference w.r.t. previous synced model.
        with kargs["timer"]("sync.compress", epoch=self.conf.epoch_):
            sync_buffer = self.compressor.compress(grads_tb)

        # sync and decompress.
        with kargs["timer"]("sync.sync_and_decompress", epoch=self.conf.epoch_):
            self.compressor.sync(sync_buffer)
            synced_updates_tb = self.compressor.decompress(sync_buffer)

        # unpack the synced info and update the consensus params.
        with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
            params_tb.buffer -= self.param_groups[0]["lr"] * synced_updates_tb.buffer
            params_tb.unpack(params)
        return sync_buffer["n_bits"]


class ExactSignCompressor(object):
    def __init__(
        self,
        rank,
        world_size,
        majority_vote,
        aggregator,
        comm_op,
        comm_device,
        use_ipc,
        **kargs
    ):
        # assign the common hyper-parameters
        self.rank = rank
        self.world_size = world_size
        self.majority_vote = majority_vote
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.use_ipc = use_ipc
        self.kargs = kargs
        self.compressor_fn = SignCompressor()

    def compress(self, grads_tb):
        # get the sign/magnitude for the tensor (to be transmitted).
        sync_buffer = dict()

        # concat the update magnitude and directions.
        signs, sign_size = self.compressor_fn.compress(grads_tb.buffer)

        # get n_bits to transmit.
        n_bits = get_n_bits(signs)

        # update shared dict.
        sync_buffer["grads_tb"] = grads_tb
        sync_buffer["signs"] = signs
        sync_buffer["sign_size"] = sign_size
        sync_buffer["n_bits"] = n_bits
        return sync_buffer

    def sync(self, sync_buffer):
        # prepare sync.
        to_sync_signs = sync_buffer["signs"]
        if self.comm_device == "cpu":
            to_sync_signs = to_sync_signs.cpu().pin_memory()

        # sync.
        synced_signs, sync_req = self.aggregator_fn._agg(
            to_sync_signs, communication_scheme="all_gather", async_op=True
        )

        # update sync_buffer.
        sync_buffer["sync_req"] = sync_req
        sync_buffer["synced_signs"] = synced_signs

    def decompress(self, sync_buffer):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_req"])

        # init placeholder.
        synced_updates_tb = deepcopy(sync_buffer["grads_tb"])
        synced_updates_tb.buffer = torch.zeros_like(synced_updates_tb.buffer)

        # decompress and update.
        for rank in range(self.world_size):
            # get signs and build its tensorbuffer.
            synced_updates_tb.buffer += self.compressor_fn.uncompress(
                comm.recover_device(
                    sync_buffer["synced_signs"][rank],
                    device=sync_buffer["grads_tb"].buffer.device,
                ),
                sync_buffer["sign_size"],
            )

        # average grad.
        if self.majority_vote:
            synced_updates_tb.buffer = torch.sign(synced_updates_tb.buffer)
        else:
            synced_updates_tb.buffer /= self.world_size * 1.0
        return synced_updates_tb
