# -*- coding: utf-8 -*-
import torch
import torch.distributed as dist

"""some auxiliary functions for communication."""


def global_average(sum, count, on_cuda=True):
    def helper(array):
        array = torch.FloatTensor(array)
        array = array.cuda() if on_cuda else array
        dist.all_reduce(array, op=dist.ReduceOp.SUM)
        all_sum, all_count = array
        if all_count == 0:
            return 0
        else:
            return all_sum / all_count

    avg = helper([sum, count])
    return avg


def elementwise_min(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return tensor


def broadcast(tensor, src):
    return dist.broadcast(tensor, src=src)


"""some aggregation functions."""


def _get_data(param_groups, idx, is_get_grad):
    # Define the function to get the data.
    # when we create the param_group, each group only has one param.
    if is_get_grad:
        return param_groups[idx]["params"][0].grad
    else:
        return param_groups[idx]["params"][0]


def _get_shape(param_groups, idx):
    return param_groups[idx]["param_size"], param_groups[idx]["nelement"]


def get_data(param_groups, param_names, is_get_grad=True):
    data, shapes = [], []
    for idx, _ in param_names:
        _data = _get_data(param_groups, idx, is_get_grad)
        if _data is not None:
            data.append(_data)
            shapes.append(_get_shape(param_groups, idx))
    return data, shapes


def flatten(tensors, shapes=None, use_cuda=True):
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    vec = torch.empty(
        pointers[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)
    return vec


def unflatten(tensors, synced_tensors, shapes):
    pointer = 0

    for tensor, shape in zip(tensors, shapes):
        param_size, nelement = shape
        tensor.data[:] = synced_tensors[pointer : pointer + nelement].view(param_size)
        pointer += nelement


"""auxiliary."""


def recover_device(data, device=None):
    if device is not None:
        return data.to(device)
    else:
        return data


"""main aggregators."""


class Aggregation(object):
    """Aggregate udpates / models from different processes."""

    def _agg(self, data, op):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        raise NotImplementedError

    def agg_model(self, model, op):
        """Aggregate models by model weight.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            param.data = self._agg(param.data, op=op)

    def agg_grad(self, model, op):
        """Aggregate models gradients.
        Args:
            model (:obj:`torch.Module`): Models to be averaged.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        """
        # Aggregate layer by layer
        for _, param in enumerate(model.parameters()):
            grad = self._agg(param.grad.data, op=op)
            param.grad.data = grad


class CentralizedAggregation(Aggregation):
    """Aggregate udpates / models from different processes."""

    def __init__(self, rank, world, neighbors_info):
        # init
        self.rank = rank

        # define the dist group.
        neighbor_ranks = list(neighbors_info.keys())
        if len(neighbor_ranks) == 0:
            self.group = None
        else:
            self.group = dist.new_group(neighbor_ranks)

        # get the world size from the view of the current rank.
        self.world_size = float(len(neighbor_ranks))

    def _agg(
        self,
        data,
        op=None,
        distributed=True,
        communication_scheme="all_reduce",
        async_op=False,
        **kargs,
    ):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        if not distributed:
            return data

        # do the real sync.
        if communication_scheme == "all_reduce":
            if op == "avg":
                req = dist.all_reduce(
                    data, op=dist.ReduceOp.SUM, group=self.group, async_op=async_op
                )
            elif op == "sum":
                req = dist.all_reduce(
                    data, op=dist.ReduceOp.SUM, group=self.group, async_op=async_op
                )
            else:
                raise NotImplementedError

            if async_op:
                # it would be dangerous to use `avg` operation with async.
                return data, req
            else:
                if op == "avg":
                    return data / self.world_size
                else:
                    return data
        elif communication_scheme == "reduce":
            if op == "sum":
                req = dist.reduce(
                    data,
                    dst=kargs["dst_rank"],
                    op=dist.ReduceOp.SUM,
                    group=self.group,
                    async_op=async_op,
                )
            else:
                raise NotImplementedError

            if async_op:
                return data, req
            else:
                if op == "sum":
                    return data
                else:
                    raise NotImplementedError
        elif communication_scheme == "all_gather":
            gathered_list = [
                torch.empty_like(data) for _ in range(int(self.world_size))
            ]
            req = dist.all_gather(
                gathered_list, data, group=self.group, async_op=async_op
            )
            if async_op:
                return gathered_list, req
            else:
                return gathered_list
        else:
            raise NotImplementedError

    def complete_wait(self, req):
        req.wait()


class DecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, rank, neighbors_info):
        # init
        self.rank = rank
        self.neighbors_info = neighbors_info
        self.neighbor_ranks = [
            neighbor_rank
            for neighbor_rank in neighbors_info.keys()
            if neighbor_rank != rank
        ]

        # get the world size from the view of the current rank.
        self.world_size = float(len(self.neighbor_ranks))

    def _agg(self, data, op, force_wait=True):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, `weighted`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        # Create some tensors to host the values from neighborhood.
        local_data = {i: torch.empty_like(data) for i in self.neighbor_ranks}
        local_data[self.rank] = data

        # async send data.
        reqs = []
        for node_rank in self.neighbor_ranks:
            reqs.append(dist.isend(tensor=local_data[self.rank], dst=node_rank))
            reqs.append(dist.irecv(tensor=local_data[node_rank], src=node_rank))

        # wait until finish.
        if force_wait:
            self.complete_wait(reqs)

            # Aggregate local_data
            if op == "avg":
                output = sum(local_data.values()) / (self.world_size + 1)
            elif op == "weighted":
                output = sum(
                    [
                        tensor * self.neighbors_info[rank]
                        for rank, tensor in local_data.items()
                    ]
                )
            elif op == "get_raw_sync_data":
                output = local_data
            else:
                raise NotImplementedError("op {} is not supported yet.".format(op))
            return output
        else:
            if op == "get_raw_sync_data":
                return reqs, local_data
            else:
                raise NotImplementedError("op {} is not supported yet.".format(op))

    def complete_wait(self, reqs):
        for req in reqs:
            req.wait()


class EfficientDecentralizedAggregation(Aggregation):
    """Aggregate updates in a decentralized manner."""

    def __init__(self, world, rank, neighbors_info, graph):
        # init
        self.rank = rank
        self.world = world
        self.graph = graph

        self.neighbors_info = neighbors_info
        self.neighbor_ranks = [
            neighbor_rank
            for neighbor_rank in neighbors_info.keys()
            if neighbor_rank != rank
        ]
        self.out_edges, self.in_edges = graph.get_edges()

    def _agg(self, data, op, force_wait=True):
        """Aggregate data using `op` operation.
        Args:
            data (:obj:`torch.Tensor`): A Tensor to be aggragated.
            op (str): Aggregation methods like `avg`, `sum`, `min`, `max`, `weighted`, etc.
        Returns:
            :obj:`torch.Tensor`: An aggregated tensor.
        """
        data = data.detach().clone()
        self.in_buffer = {i: torch.empty_like(data) for i in self.neighbor_ranks}
        self.in_buffer[self.rank] = data

        # async send data.
        out_reqs, in_reqs = [], []
        for out_edge, in_edge in zip(self.out_edges, self.in_edges):
            out_req = dist.broadcast(
                tensor=self.in_buffer[self.rank],
                src=out_edge.src,
                group=out_edge.process_group,
                async_op=True,
            )
            out_reqs.append(out_req)
            in_reqs = []
            in_req = dist.broadcast(
                tensor=self.in_buffer[in_edge.src],
                src=in_edge.src,
                group=in_edge.process_group,
                async_op=True,
            )
            in_reqs.append(in_req)
        return [out_reqs, in_reqs], self.in_buffer

    def complete_wait(self, reqs):
        out_reqs, in_reqs = reqs

        while len(out_reqs) > 0:
            req = out_reqs.pop()
            req.wait()

        while len(in_reqs) > 0:
            req = in_reqs.pop()
            req.wait()


def get_aggregators(cur_rank, world, neighbors_info, aggregator_type, graph=None):
    if "centralized" == aggregator_type:
        return CentralizedAggregation(cur_rank, world, neighbors_info)
    elif "decentralized" == aggregator_type:
        return DecentralizedAggregation(cur_rank, neighbors_info)
    elif "efficient_decentralized" == aggregator_type:
        return EfficientDecentralizedAggregation(
            world=world, rank=cur_rank, neighbors_info=neighbors_info, graph=graph
        )
    else:
        raise NotImplementedError
