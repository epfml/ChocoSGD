# -*- coding: utf-8 -*-
from abc import abstractmethod, ABC
import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs
import networkx

import torch
import torch.distributed as dist


class UndirectedGraph(ABC):
    @property
    @abstractmethod
    def n_nodes(self):
        pass

    @property
    @abstractmethod
    def n_edges(self):
        pass

    @property
    @abstractmethod
    def rho(self):
        """spectral gap: 1 - \abs{\lambda_2(W)}"""
        pass

    @property
    @abstractmethod
    def beta(self):
        """
        beta = ||I - W||_2
        """
        pass

    @property
    @abstractmethod
    def matrix(self):
        """
        Doubly stochastic mixing matrix of the graph.
        """
        pass

    @property
    @abstractmethod
    def world(self):
        """The GPU location of each rank.
            The world must be specified when running the code,
            and its size must >= # of processes.
        """
        pass

    @property
    @abstractmethod
    def rank(self):
        pass

    @property
    @abstractmethod
    def ranks(self):
        pass

    @property
    @abstractmethod
    def device(self):
        pass

    @property
    @abstractmethod
    def on_cuda(self):
        pass

    @abstractmethod
    def get_neighborhood(self, node_id):
        pass


class PhysicalLayout(UndirectedGraph):
    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        self._n_mpi_process = n_mpi_process
        self._n_sub_process = n_sub_process
        self._world = world
        self._comm_device = (
            torch.device("cpu")
            if comm_device == "cpu" or comm_device is None or not on_cuda
            else torch.device("cuda")
        )
        self._rank = rank
        self._on_cuda = on_cuda

    @property
    def device(self):
        return self.world[
            self._rank * self._n_sub_process : (self._rank + 1) * self._n_sub_process
        ]

    @property
    def on_cuda(self):
        return self._on_cuda

    @property
    def comm_device(self):
        return self._comm_device

    @property
    def rank(self):
        return self._rank

    @property
    def ranks(self):
        return list(range(self.n_nodes))

    @property
    def world(self):
        assert self._world is not None
        self._world_list = self._world.split(",")
        assert self._n_mpi_process * self._n_sub_process <= len(self._world_list)
        return [int(l) for l in self._world_list]


class CompleteGraph(PhysicalLayout):
    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        super(CompleteGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank
        )
        self._mixing_matrix = np.ones((n_mpi_process, n_mpi_process)) / n_mpi_process

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def n_edges(self):
        return self._n_mpi_process * (self._n_mpi_process - 1) / 2

    @property
    def rho(self):
        return 0

    @property
    def beta(self):
        return np.linalg.norm(
            self._mixing_matrix - np.eye((self._n_mpi_process, self._n_mpi_process)),
            ord=2,
        )

    @property
    def scaling(self):
        return len(self.get_neighborhood())

    @property
    def matrix(self):
        return self._mixing_matrix

    def get_neighborhood(self):
        """it will return a dictionary:
            key: connected (including itself) rank,
            value: mixing_matrix[rank_id, neighbor_id]
        """
        row = self._mixing_matrix[self._rank]
        return {c: v for c, v in zip(range(len(row)), row)}

    def get_peers(self):
        neighbors_info = self.get_neighborhood()
        in_peers, out_peers = [
            neighbor_rank
            for neighbor_rank in neighbors_info.keys()
            if neighbor_rank != self._rank
        ]
        return in_peers, out_peers


class RingGraph(PhysicalLayout):
    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        super(RingGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank
        )
        self._mixing_matrix, self._rho = self._compute_mixing_matrix_and_rho(
            n_mpi_process
        )

        # init some pytorch specific group configuration.
        self._make_process_group()

    def _compute_mixing_matrix_and_rho(self, n):
        assert n > 2

        # create ring matrix
        diag_rows = np.array(
            [
                [1 / 3 for _ in range(n)],
                [1 / 3 for _ in range(n)],
                [1 / 3 for _ in range(n)],
            ]
        )
        positions = [-1, 0, 1]
        mixing_matrix = sp.sparse.spdiags(diag_rows, positions, n, n).tolil()

        mixing_matrix[0, n - 1] = 1 / 3
        mixing_matrix[n - 1, 0] = 1 / 3
        mixing_matrix = mixing_matrix.tocsr()

        if n > 3:
            # Find largest real part
            eigenvalues, _ = eigs(mixing_matrix, k=2, which="LR")
            lambda2 = min(abs(i.real) for i in eigenvalues)

            # Find smallest real part
            eigenvalues, _ = eigs(mixing_matrix, k=1, which="SR")
            lambdan = eigenvalues[0].real
        else:
            eigenvals = sorted(np.linalg.eigvals(mixing_matrix.toarray()), reverse=True)
            lambda2 = eigenvals[1]
            lambdan = eigenvals[-1]

        return mixing_matrix, 1 - max(abs(lambda2), abs(lambdan))

    def _make_process_group(self):
        def _rotate_forward(r, p):
            return (r + p) % self.n_nodes

        def _rotate_backward(r, p):
            temp = r
            for _ in range(p):
                temp -= 1
                if temp < 0:
                    temp = self.n_nodes - 1
            return temp

        def _add_peers(rank, peers):
            for peer in peers:
                if peer not in self.phone_book[rank]:
                    self.phone_book[rank].append(Edge(dest=peer, src=rank))

        # init the group.
        self.phone_book = [[] for _ in range(self.n_nodes)]
        for rank in range(self.n_nodes):
            f_peer = _rotate_forward(rank, 1)
            b_peer = _rotate_backward(rank, 1)
            _add_peers(rank, [f_peer, b_peer])

    def get_edges(self):
        """ Returns the pairwise process groups between rank and the out and
            in-peers corresponding to 'self.rank'.
        """
        # get out- and in-peers using new group-indices
        out_edges = self.phone_book[self.rank]
        in_edges = []

        for group_index in range(len(out_edges)):
            for rank, edges in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                if self.rank == edges[group_index].dest:
                    in_edges.append(self.phone_book[rank][group_index])
        return out_edges, in_edges

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["phone_book"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.phone_book = dict()

    @property
    def n_edges(self):
        return self._n_mpi_process

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def rho(self):
        return self._rho

    @property
    def beta(self):
        return np.linalg.norm(
            self._mixing_matrix - np.eye((self._n_mpi_process, self._n_mpi_process)),
            ord=2,
        )

    @property
    def matrix(self):
        return self._mixing_matrix.toarray()

    @property
    def scaling(self):
        return len(self.get_neighborhood())

    def get_neighborhood(self):
        row = self._mixing_matrix.getrow(self._rank)
        _, cols = row.nonzero()
        vals = row.data
        return {int(c): v for c, v in zip(cols, vals)}


class TorusGraph(PhysicalLayout):
    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        super(TorusGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        # get proper _width and _height.
        m = int(np.sqrt(n_mpi_process))
        while True:
            if n_mpi_process % m == 0:
                n = int(n_mpi_process / m)
                break
            else:
                m -= 1

        # define the graph.
        graph = networkx.generators.lattice.grid_2d_graph(m, n, periodic=True)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray()
        for i in range(0, mixing_matrix.shape[0]):
            mixing_matrix[i][i] = 1
        mixing_matrix = mixing_matrix / 5
        return mixing_matrix

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def n_edges(self):
        raise NotImplementedError

    @property
    def rho(self):
        raise NotImplementedError

    @property
    def beta(self):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._mixing_matrix

    @property
    def scaling(self):
        return len(self.get_neighborhood())

    def get_neighborhood(self):
        row = self._mixing_matrix[self._rank]
        return {c: v for c, v in zip(range(len(row)), row) if v != 0}


class ExpanderGraph(PhysicalLayout):
    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        super(ExpanderGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        # define the graph.
        def modulo_inverse(i, p):
            for j in range(1, p):
                if (j * i) % p == 1:
                    return j

        graph = networkx.generators.classic.cycle_graph(n_mpi_process)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray()
        # for i in range(0, mixing_matrix.shape[0]):
        #     mixing_matrix[i][i] = 1
        mixing_matrix[0][0] = 1

        # connect with the inverse modulo p node.
        for i in range(1, mixing_matrix.shape[0]):
            mixing_matrix[i][modulo_inverse(i, n_mpi_process)] = 1

        mixing_matrix = mixing_matrix / 3

        return mixing_matrix

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def n_edges(self):
        raise NotImplementedError

    @property
    def rho(self):
        raise NotImplementedError

    @property
    def beta(self):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._mixing_matrix

    @property
    def scaling(self):
        return len(self.get_neighborhood())

    def get_neighborhood(self):
        row = self._mixing_matrix[self._rank]

        return {
            c: v for c, v in zip(range(len(row)), row) if (v != 0 or c == self._rank)
        }


class MargulisExpanderGraph(PhysicalLayout):
    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        super(MargulisExpanderGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        base = int(np.sqrt(n_mpi_process))
        assert (base * base) == n_mpi_process

        graph = networkx.generators.expanders.margulis_gabber_galil_graph(base)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray().astype(np.float)
        mixing_matrix[mixing_matrix > 1] = 1

        degrees = mixing_matrix.sum(axis=1)
        mixing_matrix = mixing_matrix.astype(np.float)
        for node in np.argsort(degrees)[::-1]:
            mixing_matrix[:, node][mixing_matrix[:, node] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, :][mixing_matrix[node, :] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, node] = (
                1 - np.sum(mixing_matrix[node, :]) + mixing_matrix[node, node]
            )
        return mixing_matrix

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def n_edges(self):
        raise NotImplementedError

    @property
    def rho(self):
        raise NotImplementedError

    @property
    def beta(self):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._mixing_matrix

    @property
    def scaling(self):
        return len(self.get_neighborhood())

    def get_neighborhood(self):
        row = self._mixing_matrix[self._rank]
        return {
            c: v for c, v in zip(range(len(row)), row) if (v != 0 or c == self._rank)
        }


class SocialNetworkGraph(PhysicalLayout):
    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        super(SocialNetworkGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank
        )
        assert n_mpi_process == 32
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        # define the graph.
        graph = networkx.davis_southern_women_graph()

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray().astype(np.float)

        degrees = mixing_matrix.sum(axis=1)
        for node in np.argsort(degrees)[::-1]:
            mixing_matrix[:, node][mixing_matrix[:, node] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, :][mixing_matrix[node, :] == 1] = 1.0 / degrees[node]
            mixing_matrix[node, node] = (
                1 - np.sum(mixing_matrix[node, :]) + mixing_matrix[node, node]
            )
        return mixing_matrix

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def n_edges(self):
        raise NotImplementedError

    @property
    def rho(self):
        raise NotImplementedError

    @property
    def beta(self):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._mixing_matrix

    @property
    def scaling(self):
        return len(self.get_neighborhood())

    def get_neighborhood(self):
        row = self._mixing_matrix[self._rank]

        return {
            c: v for c, v in zip(range(len(row)), row) if (v != 0 or c == self._rank)
        }


class RingExtGraph(PhysicalLayout):
    """
    Ring graph with skip connections to the most distant point in the graph.
    """

    def __init__(self, n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank):
        super(RingExtGraph, self).__init__(
            n_mpi_process, n_sub_process, world, comm_device, on_cuda, rank
        )
        self._mixing_matrix = self._define_graph(n_mpi_process)

    def _define_graph(self, n_mpi_process):
        assert n_mpi_process > 3 and n_mpi_process % 2 == 0

        # define the graph.
        graph = networkx.generators.classic.cycle_graph(n_mpi_process)

        # get the mixing matrix.
        mixing_matrix = networkx.adjacency_matrix(graph).toarray()
        for i in range(0, mixing_matrix.shape[0]):
            mixing_matrix[i][i] = 1

        # connect with the most distant node.
        for i in range(0, mixing_matrix.shape[0]):
            mixing_matrix[i][(i + n_mpi_process // 2) % n_mpi_process] = 1

        mixing_matrix = mixing_matrix / 4
        return mixing_matrix

    @property
    def n_nodes(self):
        return self._n_mpi_process

    @property
    def n_edges(self):
        raise NotImplementedError

    @property
    def rho(self):
        raise NotImplementedError

    @property
    def beta(self):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._mixing_matrix

    @property
    def scaling(self):
        return len(self.get_neighborhood())

    def get_neighborhood(self):
        row = self._mixing_matrix[self._rank]
        return {c: v for c, v in zip(range(len(row)), row) if v != 0}


class Edge(object):
    def __init__(self, dest, src):
        self.src = src
        self.dest = dest
        self.process_group = dist.new_group([src, dest])


def define_graph_topology(
    graph_topology,
    world,
    n_mpi_process,
    n_sub_process,
    comm_device,
    on_cuda,
    rank,
    **args
):
    if graph_topology == "complete":
        graph_class = CompleteGraph
    elif graph_topology == "ring":
        graph_class = RingGraph
    elif graph_topology == "torus":
        graph_class = TorusGraph
    elif graph_topology == "expander":
        graph_class = ExpanderGraph
    elif graph_topology == "margulis_expander":
        graph_class = MargulisExpanderGraph
    elif graph_topology == "social":
        graph_class = SocialNetworkGraph
    else:
        raise NotImplementedError

    graph = graph_class(
        n_mpi_process=n_mpi_process,
        n_sub_process=n_sub_process,
        world=world,
        comm_device=comm_device,
        on_cuda=on_cuda,
        rank=rank,
    )
    return graph
