# -*- coding: utf-8 -*-
"""
All-Reduce Distributed Model Wrapper
"""
import time

import torch
import torch.distributed as dist
from torch.cuda.comm import broadcast_coalesced, reduce_add_coalesced
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply


class AllReduceDataParallel(Module):
    """ Distributed AllReduce (data-parallel) model wrapper. """

    def __init__(self, module, conf):
        super(AllReduceDataParallel, self).__init__()

        # init the general config variables.
        self.graph = conf.graph
        self.distributed = conf.distributed
        self.comm_device = conf.comm_device

        # devices available locally (normally in terms of the current node).
        self.device_ids = self.graph.device
        assert len(self.device_ids) == len(set(self.device_ids))
        self.output_device = self.device_ids[0]

        # put model on output device.
        self.module = module.cuda() if conf.graph.on_cuda else module

        # prepare local intra-node all-reduce objects.
        if len(self.device_ids) > 1:
            self.broadcast_bucket_size = 10 * 1024 * 1024  # bytes
            self.nccl_reduce_bucket_size = 256 * 1024 * 1024  # bytes

            self._module_copies = replicate(self.module, self.device_ids, detach=True)

            self._module_copies[0] = self.module
            for cmodule in self._module_copies[1:]:
                for p, cp in zip(self.module.parameters(), cmodule.parameters()):
                    cp.requires_grad = p.requires_grad
        else:
            self._module_copies = [self.module]

        # register grad-reduction hooks
        self.__register_hooks()

    def forward(self, *inputs, **kwargs):
        """ Forward pass performed in parallel across all devices on node """
        # scatter inputs onto devices
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) > 1:
            # run forward pass across all devices
            self._sync_params()
            outputs = self.parallel_apply(
                self._module_copies[: len(inputs)], inputs, kwargs
            )
            return self.gather(outputs, self.output_device)
        else:
            return self.module(*inputs[0], **kwargs[0])

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[: len(replicas)]
        )

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=0)

    def _sync_params(self):
        """ Synchronoize parameters across devices (intra-node) """
        if len(self.device_ids) <= 1:
            return

        # intra-node parameter sync
        params = [p.data for p in self.module.parameters()]
        result = broadcast_coalesced(
            params, self.device_ids, self.broadcast_bucket_size
        )
        for tensors, module in zip(result[1:], self._module_copies[1:]):
            for tensor, param in zip(tensors, module.parameters()):
                with torch.no_grad():
                    param.set_(tensor)

                    # Assume we have just run the optimizer and zeroed the
                    # grads of the parameters on the root model. We need
                    # to zero the grads on all model replicas as well.
                    # This snippet is copied from torch.optim.Optimizer.
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

        # intra-node buffer sync
        buffers = [b.data for b in self.module.buffers()]
        if len(buffers) > 0:
            result = broadcast_coalesced(
                buffers, self.device_ids, self.broadcast_bucket_size
            )
            for tensors, module in zip(result[1:], self._module_copies[1:]):
                for tensor, buf in zip(tensors, module.buffers()):
                    with torch.no_grad():
                        buf.set_(tensor)

    def train(self, mode=True):
        super(AllReduceDataParallel, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)

    def eval(self):
        super(AllReduceDataParallel, self).eval()
        for module in self._module_copies[1:]:
            module.eval()

    def block(self):
        dist.barrier()

    def __register_hooks(self):
        """
        Registers gossip/all-reduce hooks in pre-forward/post-backward pass
        """
        # self.register_forward_pre_hook(self.__make_forward_pre_hook())
        self.register_backward_hook(self.__make_backward_hook())

    def __make_backward_hook(self):
        def hook(*unused):
            # reduce gradients across devices on a single machine
            if len(self.device_ids) > 1:

                # collect gradients from all copies
                all_grads = [[] for _ in range(len(self._module_copies))]
                for dev_idx, module in enumerate(self._module_copies):
                    for p in module.parameters():
                        if not p.requires_grad or p.grad is None:
                            continue
                        all_grads[dev_idx].append(p.grad.data)

                # reduce grads
                reduced_grads = reduce_add_coalesced(
                    all_grads, self.output_device, self.nccl_reduce_bucket_size
                )

                # update grads with reduced grads
                for grad, reduced in zip(all_grads[0], reduced_grads):
                    grad.copy_(reduced)

                # clear the gradients and parameters across all replicas
                for module in self._module_copies[1:]:
                    for param in module.parameters():
                        if param.requires_grad:
                            param.grad = None
                            with torch.no_grad():
                                param.set_()

        def queue_hook(*unused):
            Variable._execution_engine.queue_callback(hook)

        return queue_hook

    def communicator_warmup(self):
        """ time the all-reducde code """
        dist.barrier()
        time.sleep(5)
        dist.barrier()
