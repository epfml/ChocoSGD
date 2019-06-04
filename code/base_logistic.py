import numpy as np
from scipy.special import expit as sigmoid

from parameters import Parameters


class BaseLogistic:
    def __init__(self, params: Parameters):
        self.params = params
        self.x_estimate = None
        self.x = None

    def lr(self, epoch, iteration, num_samples, d):
        p = self.params
        t = epoch * num_samples + iteration
        if p.lr_type == 'constant':
            return p.initial_lr
        if p.lr_type == 'epoch-decay':
            return p.initial_lr * (p.epoch_decay_lr ** epoch)
        if p.lr_type == 'decay':
            return p.initial_lr / (p.regularizer * (t + p.tau))
        if p.lr_type == 'bottou':
            return p.initial_lr / (1 + p.initial_lr * p.regularizer * t)

    def loss(self, A, y):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = x.copy().mean(axis=1)
        p = self.params
        loss = np.sum(np.log(1 + np.exp(-y * (A @ x)))) / A.shape[0]
        if p.regularizer:
            loss += p.regularizer * np.square(x).sum() / 2
        return loss

    def predict(self, A):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = np.mean(x, axis=1)
        logits = A @ x
        pred = 1 * (logits >= 0.)
        return pred

    def predict_proba(self, A):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = x.mean(axis=1)
        logits = A @ x
        return sigmoid(logits)

    def score(self, A, y):
        x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.copy(x)
        x = np.mean(x, axis=1)
        logits = A @ x
        pred = 2 * (logits >= 0.) - 1
        acc = np.mean(pred == y)
        return acc

    def update_estimate(self, t):
        t = int(t)  # to avoid overflow with np.int32
        p = self.params
        if p.estimate == 'final':
            self.x_estimate = self.x
        elif p.estimate == 'mean':
            rho = 1 / (t + 1)
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
        elif p.estimate == 't+tau':
            rho = 2 * (t + p.tau) / ((1 + t) * (t + 2 * p.tau))
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho
        elif p.estimate == '(t+tau)^2':
            rho = 6 * ((t + p.tau) ** 2) / ((1 + t) * (6 * (p.tau ** 2) + t + 6 * p.tau * t + 2 * (t ** 2)))
            self.x_estimate = self.x_estimate * (1 - rho) + self.x * rho

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.params)

    def __repr__(self):
        return str(self)
