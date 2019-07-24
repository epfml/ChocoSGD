# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed

from pcode.utils.topology import define_graph_topology
from pcode.utils.sparsification import qsgd_quantize_numpy


def get_graph_topology(n_nodes, graph_topology):
    graph = define_graph_topology(
        rank=0, n_nodes=n_nodes, on_cuda=False, world=None,
        graph_topology=graph_topology)
    return graph, graph.matrix


def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def quantize(x, quantization, ratio_to_keep=None, num_levels=None):
        # quantize according to quantization function
        # x: shape(num_features, n_cores)
        if quantization == 'full':
            return x

        q = np.zeros_like(x)
        if quantization in ['qsgd-biased', 'qsgd-unbiased']:
            is_biased = (quantization == 'qsgd-biased')
            assert num_levels
            for i in range(0, q.shape[1]):
                q[:, i] = qsgd_quantize_numpy(x[:, i], num_levels, is_biased)
            return q
        elif quantization == 'top-k':
            assert ratio_to_keep
            k = int(ratio_to_keep * q.shape[0])
            for i in range(0, q.shape[1]):
                indexes = np.argsort(np.abs(x[:, i]))[::-1]
                q[indexes[:k], i] = x[indexes[:k], i]
            return q
        elif quantization in ['random-biased', 'random-unbiased']:
            assert ratio_to_keep
            k = int(ratio_to_keep * q.shape[0])
            for i in range(0, q.shape[1]):
                indexes = np.random.choice(np.arange(q.shape[0]), k, replace=False)
                q[indexes[:k], i] = x[indexes[:k], i]
            if quantization == 'random-unbiased':
                return x.shape[0] / k * q
            return q
        else:
            raise NotImplementedError


def search_gamma(X_init, W, gamma, quantization,
                 ratio_to_keep=None, num_levels=None, num_iter=1000):
    X = np.copy(X_init)
    X_hat = np.zeros_like(X)
    x_0 = np.mean(X, axis=1)
    errors = [np.sum((X.T - x_0) ** 2) + np.sum((X.T - X_hat.T) ** 2)]
    for _ in range(0, num_iter):
        X += gamma * X_hat.dot(W - np.eye(W.shape[0]))
        Q = quantize(X - X_hat, quantization, ratio_to_keep, num_levels)
        X_hat += Q
        errors += [np.sum((X.T - x_0) ** 2) + np.sum((X.T - X_hat.T) ** 2)]
    return gamma, errors[-1], errors, X


def evaluate_consensus_stepsize(model, n_nodes, graph_topology,
                                quantization_config, gamma_grid=None,
                                num_iters=1000,
                                n_jobs=48, backend='processes'):
    """The choice of config could be:
        config = {'quantization': 'qsgd-biased', 'num_levels': 16}
        config = {'quantization': 'top-k', 'ratio_to_keep': 0.1}
    """
    # Init the graph topology, define the vector to sync.
    _, mixing_matrix = get_graph_topology(n_nodes, graph_topology)
    n_params = get_n_params(model)
    X = np.random.normal(size=(n_params, n_nodes))

    # define gamma_grid if it is missing.
    if gamma_grid is None:
        gamma_grid = np.logspace(-3, 0, num=20, endpoint=True)

    # evaluate consensus stepsize
    print('=> evaluating the consensus stepsize.')
    final_errors = Parallel(n_jobs=n_jobs, prefer=backend)(
        delayed(search_gamma)(
            X, mixing_matrix, gamma=gamma,
            **quantization_config,
            num_iter=num_iters)
        for idx, gamma in enumerate(gamma_grid)
    )

    best_gamma = min(final_errors, key=lambda x: x[1])[0]

    print("=> The best gamma = {}".format(best_gamma))
    return best_gamma
