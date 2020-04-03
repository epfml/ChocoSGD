# -*- coding: utf-8 -*-
import os
import torch
import torchtext

from pcode.datasets.partition_data import DataPartitioner
from pcode.datasets.prepare_data import get_dataset


def load_data_batch(conf, _input, _target):
    """Load a mini-batch and record the loading time."""
    if conf.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    return _input, _target


def define_dataset(conf, force_shuffle=False):
    if "rnn_lm" in conf.arch:
        dataset = define_nlp_dataset(conf, force_shuffle)
    else:
        dataset = define_cv_dataset(conf, force_shuffle)
    print("Defined dataset.")
    return dataset


"""define loaders for different datasets."""
"""nlp related task."""


def define_nlp_dataset(conf, force_shuffle):
    print("create {} dataset for rank {}".format(conf.data, conf.graph.rank))
    # create dataset.
    TEXT, train, valid, _ = get_dataset(conf, conf.data, conf.data_dir)

    # Build vocb.
    # we can use some precomputed word embeddings,
    # e.g., GloVe vectors with 100, 200, and 300.
    if conf.rnn_use_pretrained_emb:
        try:
            vectors = "glove.6B.{}d".format(conf.rnn_n_hidden)
            vectors_cache = os.path.join(conf.data_dir, ".vector_cache")
        except:
            vectors, vectors_cache = None, None
    else:
        vectors, vectors_cache = None, None
    TEXT.build_vocab(train, vectors=vectors, vectors_cache=vectors_cache)

    # Partition training data.
    train_loader, _ = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=conf.batch_size * conf.graph.n_nodes,
        bptt_len=conf.rnn_bptt_len,
        device="cuda:{}".format(conf.graph.device[0]) if conf.graph.on_cuda else None,
        repeat=True,
        shuffle=force_shuffle or conf.reshuffle_per_epoch,
    )
    _, val_loader = torchtext.data.BPTTIterator.splits(
        (train, valid),
        batch_size=conf.batch_size,
        bptt_len=conf.rnn_bptt_len,
        device="cuda:{}".format(conf.graph.device[0]) if conf.graph.on_cuda else None,
        shuffle=False,
    )

    # get some stat.
    _get_nlp_data_stat(conf, train, valid, train_loader, val_loader)
    return {"TEXT": TEXT, "train_loader": train_loader, "val_loader": val_loader}


def _get_nlp_data_stat(conf, train, valid, train_loader, val_loader):
    # configure the workload for each worker.
    # Note that: the training will access to the same # of samples (w/ or w/o partition).

    # the current implementation will always partition the data.
    conf.train_word_size = len(train.examples[0].text)
    conf.valid_word_size = len(valid.examples[0].text)

    conf.num_batches_train_per_device_per_epoch = len(train_loader)
    conf.num_whole_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.num_epochs
    )
    conf.num_warmup_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.lr_warmup_epochs
    )

    # when the training is controlled by the num_iterations.
    conf.num_iterations_per_worker = conf.num_iterations // conf.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    conf.num_batches_val_per_device_per_epoch = len(val_loader)

    # define some parameters for training.
    print(
        "\nData Stat: we have {} epochs, \
         {} mini-batches per device for training. \
         {} mini-batches per device for val. \
         The batch size: {}.".format(
            conf.num_epochs,
            conf.num_batches_train_per_device_per_epoch,
            conf.num_batches_val_per_device_per_epoch,
            conf.batch_size,
        )
    )


"""cv related task."""


def define_cv_dataset(conf, force_shuffle):
    print("Create dataset: {} for rank {}.".format(conf.data, conf.graph.rank))
    train_loader = _define_cv_dataset(
        conf,
        partition_type=conf.partition_data,
        dataset_type="train",
        force_shuffle=force_shuffle,
    )
    val_loader = _define_cv_dataset(conf, partition_type=None, dataset_type="test")

    _get_cv_data_stat(conf, train_loader, val_loader)
    return {"train_loader": train_loader, "val_loader": val_loader}


def _define_cv_dataset(conf, partition_type, dataset_type, force_shuffle=False):
    """ Given a dataset, partition it. """
    dataset = get_dataset(conf, conf.data, conf.data_dir, split=dataset_type)
    batch_size = conf.batch_size
    world_size = conf.graph.n_nodes

    # determine the data to load,
    # either the whole dataset, or a subset specified by partition_type.
    if partition_type is not None and conf.distributed:
        partition_sizes = [1.0 / world_size for _ in range(world_size)]
        partition = DataPartitioner(
            conf, dataset, partition_sizes, partition_type=partition_type
        )
        data_to_load = partition.use(conf.graph.rank)
        print("Data partition: partitioned data and use subdata.")
    else:
        data_to_load = dataset
        print("Data partition: used whole data.")

    # use Dataloader.
    data_loader = torch.utils.data.DataLoader(
        data_to_load,
        batch_size=batch_size,
        shuffle=force_shuffle or dataset_type == "train",
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
    )

    print(
        (
            "Data stat: we have {} samples for {}, "
            + "load {} data for process (rank {}). "
            + "The batch size is {}, number of batches is {}."
        ).format(
            len(dataset),
            dataset_type,
            len(data_to_load),
            conf.graph.rank,
            batch_size,
            len(data_loader),
        )
    )
    return data_loader


def _get_cv_data_stat(conf, train_loader, val_loader):
    # configure the workload for each worker.
    # Note that: the training will access to the same # of samples (w/ or w/o partition).

    # when it is w/ partition, then return the true local loader size.
    # when it is w/o partition, then return the local loader size / world size.
    conf.num_batches_train_per_device_per_epoch = (
        len(train_loader) // conf.graph.n_nodes
        if conf.partition_data is None
        else len(train_loader)
    )
    conf.num_whole_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.num_epochs
    )
    conf.num_warmup_train_batches_per_worker = (
        conf.num_batches_train_per_device_per_epoch * conf.lr_warmup_epochs
    )

    # when the training is controlled by the num_iterations.
    conf.num_iterations_per_worker = conf.num_iterations // conf.graph.n_nodes

    # get the data statictics (on behalf of each worker) for val.
    conf.num_batches_val_per_device_per_epoch = len(val_loader)

    # define some parameters for training.
    print(
        "\nData Stat: we have {} epochs, \
         {} mini-batches per device for training. \
         {} mini-batches per device for val. \
         The batch size: {}.".format(
            conf.num_epochs,
            conf.num_batches_train_per_device_per_epoch,
            conf.num_batches_val_per_device_per_epoch,
            conf.batch_size,
        )
    )
