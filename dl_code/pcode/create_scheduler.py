# -*- coding: utf-8 -*-
import pcode.utils.auxiliary as auxiliary


class Scheduler(object):
    def __init__(self, conf):
        # init
        self.conf = conf
        self.local_index = 0 if "local_index" not in conf else conf.local_index
        self.init_learning_rate()
        self.init_lr_scheduler()

    def update_from_checkpoint(self, checkpoint):
        self.conf.local_index = checkpoint["local_index"]
        self.local_index = checkpoint["local_index"]
        self.conf.best_perf = checkpoint["best_perf"]

    def set_best_tracker(self, best_tracker):
        self.best_tracker = best_tracker

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != "optimizer" and "scheduler" not in key
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def init_learning_rate(self):
        # init the learning rates.
        self.conf.init_warmup_lr = self.conf.lr
        self.conf.base_batch_size = (
            self.conf.base_batch_size
            if self.conf.base_batch_size is not None
            else self.conf.batch_size
        )
        self.learning_rate_per_samples = self.conf.lr / self.conf.base_batch_size

        if self.conf.lr_scaleup:
            if self.conf.lr_scaleup_type == "linear":
                _lr = self.learning_rate_per_samples * self.conf.batch_size
                if auxiliary.is_float(self.conf.lr_scaleup_factor):
                    _scale = float(self.conf.lr_scaleup_factor)
                else:
                    if self.conf.lr_scaleup_factor == "graph":
                        _scale = self.conf.graph.scaling
                    elif self.conf.lr_scaleup_factor == "world":
                        _scale = self.conf.graph.n_nodes
                    else:
                        raise NotImplementedError
            elif self.conf.lr_scaleup_type == "sqrt":
                _lr = self.conf.lr
                _scale = (
                    1.0
                    * self.conf.graph.n_nodes
                    * self.conf.batch_size
                    / self.conf.base_batch_size
                ) ** 0.5
            else:
                raise NotImplementedError
        else:
            _lr = self.learning_rate_per_samples * self.conf.batch_size
            _scale = 1

        # get the eventual learning the backup.
        self.conf.learning_rate = _lr * _scale
        self.old_learning_rate = self.conf.learning_rate
        print(
            "learning rate will be scaled by the factor of {}. The scaled lr={}".format(
                _scale, self.conf.learning_rate
            )
        )

    def init_lr_scheduler(self):
        _lr_schedule_scheme = self.conf.lr_schedule_scheme
        if (
            _lr_schedule_scheme == "strict"
            or _lr_schedule_scheme == "custom_one_cycle"
            or _lr_schedule_scheme == "custom_multistep"
            or _lr_schedule_scheme == "custom_convex_decay"
        ):
            self.lr_scheduler = DeterministicLRScheduler(self.conf).get_lr_scheduler()
        elif _lr_schedule_scheme == "reduce_on_plateau":
            self.lr_scheduler = AdaptiveLRScheduler(self.conf).get_lr_scheduler()

    def get_lr(self, **kargs):
        return self.lr_scheduler(self.epoch_, **kargs)

    def step(self, optimizer, **kargs):
        self.update_training_progress()

        # get the new learning rate.
        lr = self.get_lr()
        if lr is None:
            lr = self.old_learning_rate

        # apply the new learning rate.
        if self.old_learning_rate != lr:
            self.old_learning_rate = lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def update_training_progress(self):
        self.local_index += 1
        self.epoch_ = (
            self.local_index / self.conf.num_batches_train_per_device_per_epoch
        )
        self.conf.local_index = self.local_index
        self.conf.epoch_ = self.epoch_
        self.epoch = int(self.epoch_)

    def is_stop(self):
        if self.conf.stop_criteria == "epoch":
            return self.epoch >= self.conf.num_epochs
        elif self.conf.stop_criteria == "iteration":
            return self.local_index >= self.conf.num_iterations_per_worker


class AdaptiveLRScheduler(object):
    def __init__(self, conf):
        self.conf = conf

    def get_lr_scheduler(self):
        def f(epoch_index, **kargs):
            pass

        return f


class DeterministicLRScheduler(object):
    def __init__(self, conf):
        self.conf = conf

    def get_lr_scheduler(self):
        epoch_fields, lr_fields, scale_indicators = self.get_scheduling_setup()
        lr_schedulers = self.build_lr_schedulers(
            epoch_fields, lr_fields, scale_indicators
        )
        print(
            "\nDefine scheduler: epoch_fields={}, lr_fields={}, lr_schedulers={}\n".format(
                epoch_fields, lr_fields, lr_schedulers
            )
        )
        return self._get_lr_scheduler(epoch_fields, lr_schedulers)

    def _get_lr_scheduler(self, epoch_fields, lr_schedulers):
        def f(epoch_index, **kargs):
            def _is_fall_in(index, left_index, right_index):
                return left_index <= index < right_index

            for ind, (epoch_left, epoch_right) in enumerate(epoch_fields):
                if _is_fall_in(epoch_index, epoch_left, epoch_right):
                    return lr_schedulers[ind](epoch_index)

        return f

    def get_scheduling_setup(self):
        if self.conf.lr_schedule_scheme == "strict":
            return _get_scheduling_setup_for_strict(self.conf)
        elif "custom_one_cycle" == self.conf.lr_schedule_scheme:
            # NOTE: The scheme yet does not support multi-GPU training.
            # No warmup and no linear scale are applied.
            return _get_scheduling_setup_for_onecycle(self.conf)
        elif "custom_multistep" == self.conf.lr_schedule_scheme:
            return _get_scheduling_setup_for_multistep(self.conf)
        elif "custom_convex_decay" == self.conf.lr_schedule_scheme:
            return _get_scheduling_setup_for_convex_decay(self.conf)
        else:
            raise NotImplementedError

    def build_lr_schedulers(self, epoch_fields, lr_fields, scale_indicators):
        lr_schedulers = dict()

        for field_id, (epoch_field, lr_field, indicator) in enumerate(
            zip(epoch_fields, lr_fields, scale_indicators)
        ):
            lr_scheduler = self._build_lr_scheduler(epoch_field, lr_field, indicator)
            lr_schedulers[field_id] = lr_scheduler
        return lr_schedulers

    def _build_lr_scheduler(self, epoch_field, lr_field, scale_indicator):
        lr_left, lr_right = lr_field
        epoch_left, epoch_right = epoch_field
        n_steps = epoch_right - epoch_left

        if scale_indicator == "linear":
            return _linear_scale(lr_left, lr_right, n_steps, epoch_left)
        elif scale_indicator == "poly":
            return _poly_scale(lr_left, lr_right, n_steps, epoch_left)
        elif scale_indicator == "convex":
            assert self.conf.lr_gamma is not None
            assert self.conf.lr_mu is not None
            assert self.conf.lr_alpha is not None
            return _convex_scale(
                self.conf.lr_gamma, self.conf.lr_mu, self.conf.lr_alpha
            )
        else:
            raise NotImplementedError


"""Define the scheduling step,
    e.g., logic of epoch_fields, lr_fields and scale_indicators.

    We should be able to determine if we only use the pure info from parser,
    or use a mixed version (the second one might be more common in practice)

    For `epoch_fields`, we define it by a string separated by ',',
    e.g., '10,20,30' to indicate different ranges.
    More precisely, previous `epoch_fields` example
    is equivalent to three different epoch ranges,
    i.e., [0, 10), [10, 20), [20, 30).

    For `lr_fields`, it is corresponding to the `epoch_fields`,
    indicating the left lr and right lr for each epoch range.

    For scale_indicators,
    it is used to define how to scale the left lr and right lr
    in the corresponding epoch range.
"""

# define the formal procedure of setting up the scheduling.


def _get_scheduling_setup(conf):
    assert conf.lr_change_epochs is not None
    assert conf.lr_fields is not None
    assert conf.lr_scale_indicators is not None

    # define lr_fields
    lr_fields = _get_lr_fields(conf.lr_fields)

    # define scale_indicators
    scale_indicators = _get_lr_scale_indicators(conf.lr_scale_indicators)

    # define epoch_fields
    epoch_fields = _get_lr_epoch_fields(conf.lr_change_epochs)

    return epoch_fields, lr_fields, scale_indicators


def _get_lr_fields(lr_fields):
    return [
        [float(_lr) for _lr in lr_field.split(",")] for lr_field in lr_fields.split("/")
    ]


def _get_lr_scale_indicators(lr_scale_indicators):
    def digital2name(x):
        return {
            "0": "linear",
            "1": "poly",
            "2": "convex",  # lr = \gamma / (\mu (t + a))
        }[x]

    return [digital2name(l) for l in lr_scale_indicators.split(",")]


def _get_lr_epoch_fields(lr_change_epochs):
    """note that the change points exclude the head and tail of the epochs.
    """
    lr_change_epochs = [int(l) for l in lr_change_epochs.split(",")]
    from_s = lr_change_epochs[:-1]
    to_s = lr_change_epochs[1:]
    return list(zip(from_s, to_s))


# case: _get scheduling setup for "strict learnign rate" configuration from the parser.


def _get_scheduling_setup_for_strict(conf):
    # define lr_fields
    conf.lr_change_epochs = "0,{original},{full}".format(
        original=conf.lr_change_epochs, full=conf.num_epochs
    )

    return _get_scheduling_setup(conf)


# case: _get scheduling setup for "onecycle learning rate" scheme.


def _get_scheduling_setup_for_onecycle(conf):
    conf.lr_fields = "{low},{high}/{high},{low}/{low},{extra_low}".format(
        low=conf.lr_onecycle_low,
        high=conf.lr_onecycle_high,
        extra_low=conf.lr_onecycle_extra_low,
    )
    conf.lr_change_epochs = "0,{half_cycle},{cycle},{full}".format(
        half_cycle=conf.lr_onecycle_num_epoch // 2,
        cycle=conf.lr_onecycle_num_epoch,
        full=conf.num_epochs,
    )
    conf.lr_scale_indicators = "0,0,0"
    return _get_scheduling_setup(conf)


# case: _get scheduling setup for "multiple-step constant learning rates" scheme.


def _get_scheduling_setup_for_multistep(conf):
    # define lr_fields
    conf.lr_fields = _build_multistep_lr_fields(
        conf.lr_change_epochs,
        conf.lr_warmup,
        conf.learning_rate,
        conf.init_warmup_lr,
        conf.lr_decay,
    )

    # define lr_change_epochs
    conf.lr_change_epochs, num_intervals = _build_multistep_lr_change_epochs(
        conf.lr_change_epochs, conf.lr_warmup, conf.lr_warmup_epochs, conf.num_epochs
    )

    # define scale_indicators
    conf.lr_scale_indicators = ",".join(["0"] * num_intervals)
    return _get_scheduling_setup(conf)


def _build_multistep_lr_fields(
    lr_change_epochs, lr_warmup, learning_rate, init_warmup_lr, lr_decay
):
    if lr_change_epochs is not None:
        _lr_fields = [
            learning_rate * ((1.0 / lr_decay) ** l)
            for l in range(len(lr_change_epochs.split(",")) + 1)
        ]
    else:
        _lr_fields = [learning_rate]

    lr_fields = "/".join(["{lr},{lr}".format(lr=lr) for lr in _lr_fields])

    if lr_warmup:
        return "{},{}/".format(init_warmup_lr, learning_rate) + lr_fields
    else:
        return lr_fields


def _build_multistep_lr_change_epochs(
    lr_change_epochs, lr_warmup, lr_warmup_epochs, num_epochs
):
    if lr_change_epochs is not None:
        lr_change_epochs = [0] + lr_change_epochs.split(",") + [num_epochs]
    else:
        lr_change_epochs = [0, num_epochs]

    if lr_warmup:
        lr_change_epochs = [0, lr_warmup_epochs] + lr_change_epochs[1:]
    return ",".join([str(x) for x in lr_change_epochs]), len(lr_change_epochs) - 1


# case: _get scheduling setup for "convex learning" scheme.


def _get_scheduling_setup_for_convex_decay(conf):
    # define lr_fields
    conf.lr_fields = "{},{}".format(conf.learning_rate, 0)

    # define lr_change_epochs
    conf.lr_change_epochs = "0,{full}".format(full=conf.num_epochs)

    # define scale_indicators
    conf.lr_scale_indicators = "2"
    return _get_scheduling_setup(conf)


"""define choice of scaling learning rate within the range."""


def _linear_scale(lr_left, lr_right, n_steps, abs_index):
    def f(index):
        step = (lr_right - lr_left) / n_steps
        return (index - abs_index) * step + lr_left

    return f


def _poly_scale(lr_left, lr_right, n_steps, abs_index):
    def f(index):
        return lr_left * ((1 - (index - abs_index) / n_steps) ** 2)

    return f


def _convex_scale(gamma, mu, alpha):
    # it is expected in the form of lr = \gamma / (\mu (t + a))
    def f(index):
        return gamma / (mu * (alpha + index))

    return f
