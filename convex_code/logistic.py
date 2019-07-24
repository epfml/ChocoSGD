import time

import numpy as np
from scipy.sparse import isspmatrix
from scipy.special import expit as sigmoid

from base_logistic import BaseLogistic
from constants import INIT_WEIGHT_STD, LOSS_PER_EPOCH
from parameters import Parameters

import networkx

def qsgd_quantize(x, d, is_biased):
    norm = np.sqrt(np.sum(np.square(x)))
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    scale = 1
    if is_biased:
        n = len(x)
        scale = 1. / (np.minimum(n / d ** 2, np.sqrt(n) / d) + 1.)
    return scale * np.sign(x) * norm * new_level / d

class LogisticDecentralizedSGD(BaseLogistic):
    """
    2 classes logistic regression on dense dataset.
    A: (num_samples, num_features)
    y: (num_features, ) 0, 1 labels
    """
    def __create_mixing_matrix(self, topology, n_cores):
        if topology == 'ring':
            W = np.zeros(shape=(n_cores, n_cores))
            value = 1./3 if n_cores >= 3 else 1./2
            np.fill_diagonal(W, value)
            np.fill_diagonal(W[1:], value, wrap=False)
            np.fill_diagonal(W[:, 1:], value, wrap=False)
            W[0, n_cores - 1] = value
            W[n_cores - 1, 0] = value
            return W
        elif topology == 'centralized':
            W = np.ones((n_cores, n_cores), dtype=np.float64) / n_cores
            return W
        elif topology == 'disconnected':
            W = np.eye(n_cores)
            return W
        else:
            print('torus topology!')
            assert topology == 'torus'
            assert int(np.sqrt(n_cores)) ** 2 == n_cores
            G = networkx.generators.lattice.grid_2d_graph(int(np.sqrt(n_cores)),
                int(np.sqrt(n_cores)), periodic=True)
            W = networkx.adjacency_matrix(G).toarray()
            for i in range(0, W.shape[0]):
                W[i][i] = 1
            W = W / 5
            return W


    def __init__(self, params: Parameters):
        super().__init__(params)
        self.x = None
        self.x_estimate = None
        self.W = self.__create_mixing_matrix(params.topology, params.n_cores)

    def __quantize(self, x):
        # quantize according to quantization function
        # x: shape(num_features, n_cores)
        if self.params.quantization in ['qsgd-biased', 'qsgd-unbiased']:
            is_biased = (self.params.quantization == 'qsgd-biased')
            assert self.params.num_levels
            q = np.zeros_like(x)
            for i in range(0, q.shape[1]):
                q[:, i] = qsgd_quantize(x[:, i], self.params.num_levels, is_biased)
            return q
        if self.params.quantization == 'full':
            return x
        if self.params.quantization == 'top':
            q = np.zeros_like(x)
            k = self.params.coordinates_to_keep
            for i in range(0, q.shape[1]):
                indexes = np.argsort(np.abs(x[:, i]))[::-1]
                q[indexes[:k], i] = x[indexes[:k], i]
            return q

        assert self.params.quantization in ['random-biased', 'random-unbiased']
        Q = np.zeros_like(x)
        k = self.params.coordinates_to_keep
        for i in range(0, Q.shape[1]):
            indexes = np.random.choice(np.arange(Q.shape[0]), k, replace=False)
            Q[indexes[:k], i] = x[indexes[:k], i]
        if self.params.quantization == 'random-unbiased':
            return x.shape[0] / k * Q
        return Q


    def fit(self, A, y_init):
        y = np.copy(y_init)
        num_samples, num_features = A.shape
        p = self.params

        losses = np.zeros(p.num_epoch + 1)

        # Initialization of parameters
        if self.x is None:
            self.x = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))
            self.x = np.tile(self.x, (p.n_cores, 1)).T
            self.x_estimate = np.copy(self.x)
            self.x_hat = np.copy(self.x)
            if p.method == 'old':
                self.h = np.zeros_like(self.x)
                alpha = 1. / (A.shape[1] / p.coordinates_to_keep + 1)

        # splitting data onto machines
        if p.distribute_data:
            np.random.seed(p.split_data_random_seed)
            num_samples_per_machine = num_samples // p.n_cores
            if p.split_data_strategy == 'random':
                all_indexes = np.arange(num_samples)
                np.random.shuffle(all_indexes)
            elif p.split_data_strategy == 'naive':
                all_indexes = np.arange(num_samples)
            elif p.split_data_strategy == 'label-sorted':
                all_indexes = np.argsort(y)

            indices = []
            for machine in range(0, p.n_cores - 1):
                indices += [all_indexes[num_samples_per_machine * machine:\
                                        num_samples_per_machine * (machine + 1)]]
            indices += [all_indexes[num_samples_per_machine * (p.n_cores - 1):]]
            print("length of indices:", len(indices))
            print("length of last machine indices:", len(indices[-1]))
        else:
            num_samples_per_machine = num_samples
            indices = np.tile(np.arange(num_samples), (p.n_cores, 1))
        # should have shape (num_machines, num_samples)

        # if cifar10 or mnist dataset, then make it binary
        if len(np.unique(y)) > 2:
            y[y < 5] = -1
            y[y >= 5] = 1
        print("Number of different labels:", len(np.unique(y)))
        # epoch 0 loss evaluation
        losses[0] = self.loss(A, y)

        compute_loss_every = int(num_samples_per_machine / LOSS_PER_EPOCH)
        all_losses = np.zeros(int(num_samples_per_machine * p.num_epoch / compute_loss_every) + 1)

        train_start = time.time()
        np.random.seed(p.random_seed)

        for epoch in np.arange(p.num_epoch):
            for iteration in range(num_samples_per_machine):
                t = epoch * num_samples_per_machine + iteration
                if t % compute_loss_every == 0:
                    loss = self.loss(A, y)
                    print('{} t {} epoch {} iter {} loss {} elapsed {}s'.format(p, t,
                        epoch, iteration, loss, time.time() - train_start))
                    all_losses[t // compute_loss_every] = loss
                    if np.isinf(loss) or np.isnan(loss):
                        print("finish trainig")
                        break

                lr = self.lr(epoch, iteration, num_samples_per_machine, num_features)

                # Gradient step
                x_plus = np.zeros_like(self.x)
                for machine in range(0, p.n_cores):
                    sample_idx = np.random.choice(indices[machine])
                    a = A[sample_idx]
                    x = self.x[:, machine]

                    minus_grad = y[sample_idx] * a * sigmoid(-y[sample_idx] * a.dot(x).squeeze())
                    if isspmatrix(a):
                        minus_grad = minus_grad.toarray().squeeze(0)
                    if p.regularizer:
                        minus_grad -= p.regularizer * x
                    x_plus[:, machine] = lr * minus_grad

                # Communication step
                if p.method == "plain":
                    self.x = (self.x + x_plus).dot(self.W)
                if p.method == "choco":
                    x_plus += self.x
                    self.x = x_plus + p.consensus_lr * self.x_hat.dot(self.W - np.eye(p.n_cores))
                    quantized = self.__quantize(self.x - self.x_hat)
                    self.x_hat += quantized
                elif p.method == 'dcd-psgd':
                    x_plus += self.x.dot(self.W)
                    quantized = self.__quantize(x_plus - self.x)
                    self.x += quantized
                elif p.method == 'ecd-psgd':
                    x_plus += self.x_hat.dot(self.W)
                    z = (1 - 0.5 * (t + 1)) * self.x + 0.5 * (t + 1) * x_plus
                    quantized = self.__quantize(z)
                    self.x = np.copy(x_plus)
                    self.x_hat = (1 - 2. / (t + 1)) * self.x_hat + 2./(t + 1) * quantized

                self.update_estimate(t)

            losses[epoch + 1] = self.loss(A, y)
            print("epoch {}: loss {} score {}".format(epoch, losses[epoch + 1], self.score(A, y)))
            if np.isinf(losses[epoch + 1]) or np.isnan(losses[epoch + 1]):
                print("finish trainig")
                break

        print("Training took: {}s".format(time.time() - train_start))

        return losses, all_losses
