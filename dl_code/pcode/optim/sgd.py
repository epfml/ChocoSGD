# -*- coding: utf-8 -*-
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import get_n_bits
from pcode.utils.tensor_buffer import TensorBuffer


class SGD(Optimizer):
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
        super(SGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()

        # define the aggregator.
        self.decentralized_aggregator = comm.get_aggregators(
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

        # define reducer.
        self.backend = conf.backend

        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None, **kargs):
        if self.conf.is_centralized:
            with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
                # Get data.
                grads, _ = comm.get_data(
                    self.param_groups, self.param_names, is_get_grad=True
                )
                flatten_grads = TensorBuffer(grads)

            with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
                # Aggregate the gradients.
                flatten_grads.buffer = self.world_aggregator._agg(
                    flatten_grads.buffer, op="avg", distributed=self.conf.distributed
                )

            with kargs["timer"]("sync.unflatten_grad", epoch=self.conf.epoch_):
                # unflatten grads.
                flatten_grads.unpack(grads)

            with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
                utils.apply_gradient(
                    self.param_groups, self.state, apply_grad_to_model=True
                )

            # Get n_bits to transmit.
            n_bits = get_n_bits(flatten_grads.buffer)
        else:
            with kargs["timer"]("sync.apply_grad", epoch=self.conf.epoch_):
                utils.apply_gradient(
                    self.param_groups, self.state, apply_grad_to_model=True
                )

            with kargs["timer"]("sync.get_data", epoch=self.conf.epoch_):
                # first get and flatten all params.
                params, _ = comm.get_data(
                    self.param_groups, self.param_names, is_get_grad=False
                )
                flatten_params = TensorBuffer(params)

            with kargs["timer"]("sync.sync", epoch=self.conf.epoch_):
                # prepare the sync.
                if self.conf.comm_device == "cpu":
                    flatten_params.buffer.cpu().detach_()

                # then sync.
                flatten_params.buffer = self.decentralized_aggregator._agg(
                    flatten_params.buffer, op="weighted"
                )

            with kargs["timer"]("sync.update_model", epoch=self.conf.epoch_):
                # finally unflatten.
                flatten_params.unpack(params)

            # Get n_bits to transmit.
            n_bits = get_n_bits(flatten_params.buffer)
        return n_bits
