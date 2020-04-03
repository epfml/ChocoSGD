# -*- coding: utf-8 -*-
import gc
import copy
import numpy as np
import torch
import torch.distributed as dist

from pcode.create_dataset import define_dataset, load_data_batch, _define_cv_dataset

from pcode.utils.checkpoint import save_to_checkpoint
from pcode.utils.logging import (
    display_training_stat,
    display_test_stat,
    dispaly_best_test_stat,
)
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.utils.error_handler as error_handler
import pcode.utils.auxiliary as auxiliary

# sys.excepthook = error_handler.global_except_hook


def train_and_validate(
    conf, model, criterion, scheduler, optimizer, metrics, data_loader
):
    print("=>>>> start training and validation.\n")

    # define runtime stat tracker and start the training.
    tracker_tr = RuntimeTracker(
        metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
    )

    # get the timer.
    timer = conf.timer

    # break until finish expected full epoch training.
    print("=>>>> enter the training.\n")
    while True:
        dist.barrier()

        # configure local step.
        for _input, _target in data_loader["train_loader"]:
            model.train()
            scheduler.step(optimizer)

            # load data
            with timer("load_data", epoch=scheduler.epoch_):
                _input, _target = load_data_batch(conf, _input, _target)

            # inference and get current performance.
            with timer("forward_pass", epoch=scheduler.epoch_):
                optimizer.zero_grad()
                loss = inference(model, criterion, metrics, _input, _target, tracker_tr)

            with timer("backward_pass", epoch=scheduler.epoch_):
                loss.backward()

            with timer("sync_complete", epoch=scheduler.epoch_):
                n_bits_to_transmit = optimizer.step(timer=timer, scheduler=scheduler)

            # display the logging info.
            display_training_stat(conf, scheduler, tracker_tr, n_bits_to_transmit)

            # finish one epoch training and to decide if we want to val our model.
            if scheduler.epoch_ % 1 == 0:
                if tracker_tr.stat["loss"].avg > 1e3 or np.isnan(
                    tracker_tr.stat["loss"].avg
                ):
                    print("\nThe process diverges!!!!!Early stop it.")
                    error_handler.abort()

                # each worker finish one epoch training.
                do_validate(
                    conf, model, optimizer, criterion, scheduler, metrics, data_loader
                )

                # refresh the logging cache at the begining of each epoch.
                tracker_tr.reset()

                # evaluate (and only inference) on the whole training loader.
                if (
                    conf.evaluate_consensus or scheduler.is_stop()
                ) and not conf.train_fast:
                    # prepare the dataloader for the consensus evaluation.
                    _data_loader = {
                        "val_loader": _define_cv_dataset(
                            conf,
                            partition_type=None,
                            dataset_type="train",
                            force_shuffle=True,
                        )
                    }

                    # evaluate on the local model.
                    conf.logger.log("eval the local model on full training data.")
                    validate(
                        conf,
                        model,
                        optimizer,
                        criterion,
                        scheduler,
                        metrics,
                        data_loader=_data_loader,
                        label="eval_local_model_on_full_training_data",
                        force_evaluate_on_averaged_model=False,
                    )

                    # evaluate on the averaged model.
                    conf.logger.log("eval the averaged model on full training data.")
                    copied_model = copy.deepcopy(
                        model.module
                        if "DataParallel" == model.__class__.__name__
                        else model
                    )
                    optimizer.world_aggregator.agg_model(copied_model, op="avg")
                    validate(
                        conf,
                        copied_model,
                        optimizer,
                        criterion,
                        scheduler,
                        metrics,
                        data_loader=_data_loader,
                        label="eval_averaged_model_on_full_training_data",
                        force_evaluate_on_averaged_model=False,
                    )

                # determine if the training is finished.
                if scheduler.is_stop():
                    # save json.
                    conf.logger.save_json()

                    # temporarily hack the exit parallelchoco
                    if optimizer.__class__.__name__ == "ParallelCHOCO":
                        error_handler.abort()
                    return

            # display tracking time.
            if (
                conf.graph.rank == 0
                and conf.display_tracked_time
                and scheduler.local_index % conf.summary_freq == 0
            ):
                print(timer.summary())

        # reshuffle the data.
        if conf.reshuffle_per_epoch:
            print("\nReshuffle the dataset.")
            del data_loader
            gc.collect()
            data_loader = define_dataset(conf)


def inference(model, criterion, metrics, _input, _target, tracker=None):
    """Inference on the given model and get loss and accuracy."""
    output = model(_input)
    loss = criterion(output, _target)
    performance = metrics.evaluate(loss, output, _target)
    if tracker is not None:
        tracker.update_metrics([loss.item()] + performance, n_samples=_input.size(0))
    return loss


def do_validate(conf, model, optimizer, criterion, scheduler, metrics, data_loader):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    print("Enter validation phase.")
    performance = validate(
        conf, model, optimizer, criterion, scheduler, metrics, data_loader
    )

    # remember best performance and display the val info.
    scheduler.best_tracker.update(performance[0], scheduler.epoch_)
    dispaly_best_test_stat(conf, scheduler)

    # save to the checkpoint.
    if not conf.train_fast:
        save_to_checkpoint(
            conf,
            {
                "arch": conf.arch,
                "current_epoch": scheduler.epoch,
                "local_index": scheduler.local_index,
                "best_perf": scheduler.best_tracker.best_perf,
                "optimizer": optimizer.state_dict(),
                "state_dict": model.state_dict(),
            },
            scheduler.best_tracker.is_best,
            dirname=conf.checkpoint_dir,
            filename="checkpoint.pth.tar",
            save_all=conf.save_all_models,
        )
    print("Finished validation.")


def validate(
    conf,
    model,
    optimizer,
    criterion,
    scheduler,
    metrics,
    data_loader,
    label="local_model",
    force_evaluate_on_averaged_model=True,
):
    """A function for model evaluation."""

    def _evaluate(_model, label):
        # define stat.
        tracker_te = RuntimeTracker(
            metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
        )

        # switch to evaluation mode
        _model.eval()

        for _input, _target in data_loader["val_loader"]:
            # load data and check performance.
            _input, _target = load_data_batch(conf, _input, _target)

            with torch.no_grad():
                inference(_model, criterion, metrics, _input, _target, tracker_te)

        # display the test stat.
        display_test_stat(conf, scheduler, tracker_te, label)

        # get global (mean) performance
        global_performance = tracker_te.evaluate_global_metrics()
        return global_performance

    # evaluate the averaged local model on the validation dataset.
    if (
        conf.graph_topology != "complete"
        and not conf.train_fast
        and conf.evaluate_consensus
        and force_evaluate_on_averaged_model
    ):
        copied_model = copy.deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        optimizer.world_aggregator.agg_model(copied_model, op="avg")
        _evaluate(copied_model, label="averaged_model")

        # get the l2 distance of the local model to the averaged model
        conf.logger.log_metric(
            name="stat",
            values={
                "rank": conf.graph.rank,
                "epoch": scheduler.epoch_,
                "distance": auxiliary.get_model_difference(model, copied_model),
            },
            tags={"split": "test", "type": "averaged_model"},
        )

    # evaluate each local model on the validation dataset.
    global_performance = _evaluate(model, label=label)
    return global_performance
