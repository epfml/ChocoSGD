# -*- coding: utf-8 -*-
import os
import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from parameters import get_args

import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.create_scheduler as create_scheduler

import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.op_paths as op_paths
import pcode.utils.stat_tracker as stat_tracker
import pcode.utils.logging as logging
from pcode.utils.timer import Timer


def init_distributed_world(conf, backend):
    if backend == "mpi":
        dist.init_process_group("mpi")
    elif backend == "nccl" or backend == "gloo":
        # init the process group.
        _tmp_path = os.path.join(conf.checkpoint, "tmp", conf.timestamp)
        op_paths.build_dirs(_tmp_path)

        dist_init_file = os.path.join(_tmp_path, "dist_init")

        torch.distributed.init_process_group(
            backend=backend,
            init_method="file://" + os.path.abspath(dist_init_file),
            timeout=datetime.timedelta(seconds=120),
            world_size=conf.n_mpi_process,
            rank=conf.local_rank,
        )
    else:
        raise NotImplementedError


def main(conf):
    try:
        init_distributed_world(conf, backend=conf.backend)
        conf.distributed = True and conf.n_mpi_process > 1
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    init_config(conf)

    # define the timer for different operations.
    # if we choose the `train_fast` mode, then we will not track the time.
    conf.timer = Timer(
        verbosity_level=1 if conf.track_time and not conf.train_fast else 0,
        log_fn=conf.logger.log_metric,
        on_cuda=conf.on_cuda,
    )

    # create dataset.
    data_loader = create_dataset.define_dataset(conf, force_shuffle=True)

    # create model
    model = create_model.define_model(conf, data_loader=data_loader)

    # define the optimizer.
    optimizer = create_optimizer.define_optimizer(conf, model)

    # define the lr scheduler.
    scheduler = create_scheduler.Scheduler(conf, optimizer)

    # add model with data-parallel wrapper.
    if conf.graph.on_cuda:
        if conf.n_sub_process > 1:
            model = torch.nn.DataParallel(model, device_ids=conf.graph.device)

    # (optional) reload checkpoint
    try:
        checkpoint.maybe_resume_from_checkpoint(conf, model, optimizer, scheduler)
    except RuntimeError as e:
        conf.logger.log(f"Resume Error: {e}")
        conf.resumed = False

    # train amd evaluate model.
    if "rnn_lm" in conf.arch:
        from pcode.distributed_running_nlp import train_and_validate

        # safety check.
        assert (
            conf.n_sub_process == 1
        ), "our current data-parallel wrapper does not support RNN."

        # define the criterion and metrics.
        criterion = nn.CrossEntropyLoss(reduction="mean")
        criterion = criterion.cuda() if conf.graph.on_cuda else criterion
        metrics = create_metrics.Metrics(
            model.module if "DataParallel" == model.__class__.__name__ else model,
            task="language_modeling",
        )

        # define the best_perf tracker, either empty or from the checkpoint.
        best_tracker = stat_tracker.BestPerf(
            best_perf=None if "best_perf" not in conf else conf.best_perf,
            larger_is_better=False,
        )
        scheduler.set_best_tracker(best_tracker)

        # get train_and_validate_func
        train_and_validate_fn = train_and_validate
    else:
        from pcode.distributed_running_cv import train_and_validate

        # define the criterion and metrics.
        criterion = nn.CrossEntropyLoss(reduction="mean")
        criterion = criterion.cuda() if conf.graph.on_cuda else criterion
        metrics = create_metrics.Metrics(
            model.module if "DataParallel" == model.__class__.__name__ else model,
            task="classification",
        )

        # define the best_perf tracker, either empty or from the checkpoint.
        best_tracker = stat_tracker.BestPerf(
            best_perf=None if "best_perf" not in conf else conf.best_perf,
            larger_is_better=True,
        )
        scheduler.set_best_tracker(best_tracker)

        # get train_and_validate_func
        train_and_validate_fn = train_and_validate

    # save arguments to disk.
    checkpoint.save_arguments(conf)

    # start training.
    train_and_validate_fn(
        conf,
        model=model,
        criterion=criterion,
        scheduler=scheduler,
        optimizer=optimizer,
        metrics=metrics,
        data_loader=data_loader,
    )


def init_config(conf):
    # define the graph for the computation.
    cur_rank = dist.get_rank() if conf.distributed else 0
    conf.graph = topology.define_graph_topology(
        graph_topology=conf.graph_topology,
        world=conf.world,
        n_mpi_process=conf.n_mpi_process,  # the # of total main processes.
        n_sub_process=conf.n_sub_process,  # the # of subprocess for each main process.
        comm_device=conf.comm_device,
        on_cuda=conf.on_cuda,
        rank=cur_rank,
    )
    conf.is_centralized = conf.graph_topology == "complete"

    # re-configure batch_size if sub_process > 1.
    if conf.n_sub_process > 1:
        conf.batch_size = conf.batch_size * conf.n_sub_process

    # configure cuda related.
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.manual_seed(conf.manual_seed)
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.set_device(conf.graph.device[0])
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True if conf.train_fast else False

    # define checkpoint for logging.
    checkpoint.init_checkpoint(conf)

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    logging.display_args(conf)


if __name__ == "__main__":
    # parse the arguments.
    conf = get_args()

    # configure for multi-process training.
    if conf.optimizer == "parallel_choco":
        mp.set_start_method("forkserver", force=True)
        # mp.set_start_method("spawn", force=True)
        mp.set_sharing_strategy("file_system")

    # enter the training procedure.
    main(conf)
