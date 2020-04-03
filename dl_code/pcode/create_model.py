# -*- coding: utf-8 -*-

import torch.distributed as dist

import pcode.models as models


def define_model(conf, **kargs):
    if "rnn_lm" in conf.arch:
        return define_nlp_model(conf, TEXT=kargs["data_loader"]["TEXT"])
    else:
        return define_cv_model(conf)


"""define loaders for different models."""


def define_cv_model(conf):
    if "wideresnet" in conf.arch:
        model = models.__dict__["wideresnet"](conf)
    elif "resnet" in conf.arch:
        model = models.__dict__["resnet"](conf)
    elif "densenet" in conf.arch:
        model = models.__dict__["densenet"](conf)
    elif "vgg" in conf.arch:
        model = models.__dict__["vgg"](conf)
    elif "lenet" in conf.arch:
        model = models.__dict__["lenet"](conf)
    else:
        model = models.__dict__[conf.arch](conf)

    if conf.graph.on_cuda:
        model = model.cuda()

    # get a consistent init model over the world.
    if conf.distributed:
        consistent_model(conf, model)

    # get the model stat info.
    get_model_stat(conf, model)
    return model


def define_nlp_model(conf, TEXT):
    print("=> creating model '{}'".format(conf.arch))

    # get embdding size and num_tokens.
    weight_matrix = TEXT.vocab.vectors

    if weight_matrix is not None:
        conf.n_tokens, emb_size = weight_matrix.size(0), weight_matrix.size(1)
    else:
        conf.n_tokens, emb_size = len(TEXT.vocab), conf.rnn_n_hidden

    # create model.
    model = models.RNNLM(
        ntoken=conf.n_tokens,
        ninp=emb_size,
        nhid=conf.rnn_n_hidden,
        nlayers=conf.rnn_n_layers,
        tie_weights=conf.rnn_tie_weights,
        dropout=conf.drop_rate,
        weight_norm=conf.rnn_weight_norm,
    )

    # init the model.
    if weight_matrix is not None:
        model.encoder.weight.data.copy_(weight_matrix)

    if conf.graph.on_cuda:
        model = model.cuda()

    # consistent the model.
    consistent_model(conf, model)
    get_model_stat(conf, model)
    return model


"""some utilities functions."""


def get_model_stat(conf, model):
    print(
        "=> creating model '{}. total params for process {}: {}M".format(
            conf.arch,
            conf.graph.rank,
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
    )


def consistent_model(conf, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    print("consistent model for process (rank {})".format(conf.graph.rank))
    cur_rank = conf.graph.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
