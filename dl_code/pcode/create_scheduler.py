# -*- coding: utf-8 -*-
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pcode.utils.auxiliary as auxiliary


class Scheduler(object):
    def __init__(self, conf, optimizer):
        # init
        self.conf = conf
        self.local_index = 0
        self.optimizer = optimizer
        self._update_training_progress()
        self.init_learning_rate()
        self.init_lr_scheduler()

    def init_learning_rate(self):
        # determine the learning_rate_per_samples.
        self.lr_scaleup_init_lr = (
            self.conf.lr_scaleup_init_lr
            if self.conf.lr_scaleup_init_lr is not None
            else self.conf.lr
        )
        self.conf.base_batch_size = (
            self.conf.base_batch_size
            if self.conf.base_batch_size is not None
            else self.conf.batch_size
        )
        self.learning_rate_per_samples = self.conf.lr / self.conf.base_batch_size
        self.learning_rate_ = self.learning_rate_per_samples * self.conf.batch_size

        # if scaleup.
        if self.conf.lr_scaleup:
            if self.conf.lr_scaleup_factor is None:
                self.lr_scaleup_factor = self.conf.graph.n_nodes
            else:
                if auxiliary.is_float(self.conf.lr_scaleup_factor):
                    self.lr_scaleup_factor = float(self.conf.lr_scaleup_factor)
                else:
                    if self.conf.lr_scaleup_factor == "graph":
                        self.lr_scaleup_factor = self.conf.graph.scaling
                    elif self.conf.lr_scaleup_factor == "world":
                        self.lr_scaleup_factor = self.conf.graph.n_nodes
                    else:
                        raise NotImplementedError

            self.learning_rate = self.learning_rate_ * self.lr_scaleup_factor
        else:
            self.learning_rate = self.learning_rate_

        # overwrite lr_scaleup_factor.
        self.lr_scaleup_factor = self.learning_rate / self.lr_scaleup_init_lr
        self.is_scaledup = True if self.lr_scaleup_factor != 1 else False

        # if warmup.
        if self.conf.lr_warmup_epochs is None:
            self.conf.lr_warmup_epochs = min(
                self.conf.lr_scaleup_factor, self.conf.lr_warmup_epochs_upper_bound
            )

        # check the warmup status.
        self.is_warmuped = (
            True if self.conf.lr_scaleup_factor != 1 and self.conf.lr_warmup else False
        )

        # update the lr for the optimizer.
        if self.is_warmuped:
            self.update_lr(self.lr_scaleup_init_lr)
        elif self.is_scaledup:
            self.update_lr(self.learning_rate)

        self.conf.logger.log(
            f"LR initialization (lr={self.conf.lr} for mini-batch size={self.conf.base_batch_size} and scaled to {self.learning_rate_} for local mini-batch size={self.conf.batch_size}): lr scaleup={self.is_scaledup}, lr warmup={self.is_warmuped}, learning_rate={self.learning_rate}."
        )

    def init_lr_scheduler(self):
        if self.conf.lr_scheduler == "MultiStepLR":
            if self.conf.lr_milestones is not None:
                milestones = [int(x) for x in self.conf.lr_milestones.split(",")]
            else:
                milestones = [self.conf.num_epochs + 1]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=milestones, gamma=self.conf.lr_decay
            )
            scheduler_info = f"use MultiStepLR scheduler: milestones={milestones}, decay_factor={self.conf.lr_decay}"
        elif self.conf.lr_scheduler == "ExponentialLR":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=self.conf.lr_decay
            )
            scheduler_info = (
                f"use ExponentialLR scheduler: decay_factor={self.conf.lr_decay}"
            )
        elif self.conf.lr_scheduler == "ReduceLROnPlateau":
            raise NotImplementedError("not support ReduceLROnPlateau yet.")
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     self.optimizer,
            #     factor=self.conf.lr_decay,
            #     mode="min",
            #     patience=self.conf.lr_patience,
            # )
        else:
            raise NotImplementedError(
                f"we do not support this scheduler={self.conf.lr_scheduler} yet."
            )

        # in case we need to warmup the learning rate scheduler.
        if self.is_warmuped:
            self.lr_scheduler = GradualWarmupScheduler(
                optimizer=self.optimizer,
                multiplier=self.lr_scaleup_factor,
                total_epoch=self.conf.lr_warmup_epochs,
                after_scheduler=lr_scheduler,
            )
            warmup_info = f"first warmup lr={self.lr_scaleup_init_lr} with factor={self.lr_scaleup_factor} from {self.lr_scaleup_init_lr} to {self.learning_rate} for {self.conf.lr_warmup_epochs} epochs, then "
        else:
            self.lr_scheduler = lr_scheduler
            warmup_info = f"first set lr={self.learning_rate}, then "
        self.conf.logger.log(
            f"LR scheduler in a nutshell: {warmup_info}{scheduler_info}."
        )

    def set_best_tracker(self, best_tracker):
        self.best_tracker = best_tracker

    def step(self, **kargs):
        self.update_training_progress()
        self.lr_scheduler.step(epoch=self.epoch_)

    def update_training_progress(self):
        self.local_index += 1
        self._update_training_progress()

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _update_training_progress(self):
        self.epoch_ = (
            self.local_index / self.conf.num_batches_train_per_device_per_epoch
        )
        self.conf.local_index = self.local_index
        self.conf.epoch_ = self.epoch_
        self.epoch = int(self.epoch_)
        self.conf.epoch = self.epoch

    def is_stop(self):
        if self.conf.stop_criteria == "epoch":
            return self.epoch >= self.conf.num_epochs
        elif self.conf.stop_criteria == "iteration":
            return self.local_index >= self.conf.num_iterations_per_worker

    def update_from_checkpoint(self, checkpoint):
        self.conf.local_index = checkpoint["local_index"]
        self.local_index = checkpoint["local_index"]
        self.conf.best_perf = checkpoint["best_perf"]


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning.

        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler is not None:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
