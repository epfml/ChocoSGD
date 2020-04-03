# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join
import argparse

import pcode.models as models
from pcode.utils.checkpoint import get_checkpoint_folder_name


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")

    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )

    # feed them to the parser.
    parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")

    # add arguments.
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)

    # dataset.
    parser.add_argument("--data", default="cifar10", help="a specific dataset name")
    parser.add_argument(
        "--data_dir", default=RAW_DATA_DIRECTORY, help="path to dataset"
    )
    parser.add_argument(
        "--use_lmdb_data",
        default=False,
        type=str2bool,
        help="use sequential lmdb dataset for better loading.",
    )
    parser.add_argument(
        "--partition_data",
        default=None,
        type=str,
        help="decide if each worker will access to all data.",
    )
    parser.add_argument("--pin_memory", default=True, type=str2bool)

    # model
    parser.add_argument(
        "--arch",
        "-a",
        default="resnet20",
        help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
    )

    # training and learning scheme
    parser.add_argument("--train_fast", type=str2bool, default=False)
    parser.add_argument("--stop_criteria", type=str, default="epoch")
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--num_iterations", type=int, default=32000)

    parser.add_argument("--avg_model", type=str2bool, default=False)
    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    parser.add_argument(
        "--batch_size",
        "-b",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument("--base_batch_size", default=None, type=int)

    # learning rate scheme
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_schedule_scheme", type=str, default=None)

    parser.add_argument("--lr_change_epochs", type=str, default=None)
    parser.add_argument("--lr_fields", type=str, default=None)
    parser.add_argument("--lr_scale_indicators", type=str, default=None)

    parser.add_argument("--lr_scaleup", type=str2bool, default=False)
    parser.add_argument("--lr_scaleup_type", type=str, default="linear")
    parser.add_argument(
        "--lr_scaleup_factor",
        type=str,
        default="graph",
        help="scale by the graph connection, or the world size",
    )
    parser.add_argument("--lr_warmup", type=str2bool, default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_decay", type=float, default=10)

    parser.add_argument("--lr_onecycle_low", type=float, default=0.15)
    parser.add_argument("--lr_onecycle_high", type=float, default=3)
    parser.add_argument("--lr_onecycle_extra_low", type=float, default=0.0015)
    parser.add_argument("--lr_onecycle_num_epoch", type=int, default=46)

    parser.add_argument("--lr_gamma", type=float, default=None)
    parser.add_argument("--lr_mu", type=float, default=None)
    parser.add_argument("--lr_alpha", type=float, default=None)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd")

    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    # the topology of the decentralized network.
    parser.add_argument("--graph_topology", default="complete", type=str)

    # compression scheme.
    parser.add_argument("--comm_algo", default=None, type=str)
    parser.add_argument(
        "--comm_op",
        default=None,
        type=str,
        choices=["compress_top_k", "compress_random_k", "quantize_qsgd", "sign"],
    )
    parser.add_argument("--compress_ratio", default=None, type=float)
    parser.add_argument(
        "--compress_warmup_values", default="0.75,0.9375,0.984375,0.996,0.999", type=str
    )
    parser.add_argument("--compress_warmup_epochs", default=0, type=int)
    parser.add_argument("--quantize_level", default=None, type=int)
    parser.add_argument("--is_biased", default=False, type=str2bool)
    parser.add_argument("--majority_vote", default=False, type=str2bool)

    parser.add_argument("--consensus_stepsize", default=0.9, type=float)
    parser.add_argument("--evaluate_consensus", default=False, type=str2bool)

    parser.add_argument("--mask_momentum", default=False, type=str2bool)
    parser.add_argument("--clip_grad", default=False, type=str2bool)
    parser.add_argument("--clip_grad_val", default=None, type=float)

    parser.add_argument("--local_step", default=1, type=int)
    parser.add_argument("--turn_on_local_step_from", default=0, type=int)
    parser.add_argument("--local_adam_memory_treatment", default=None, type=str)

    # momentum scheme
    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    # regularization
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.0, type=float)

    # configuration for different models.
    parser.add_argument("--densenet_growth_rate", default=12, type=int)
    parser.add_argument("--densenet_bc_mode", default=False, type=str2bool)
    parser.add_argument("--densenet_compression", default=0.5, type=float)

    parser.add_argument("--wideresnet_widen_factor", default=4, type=int)

    parser.add_argument("--rnn_n_hidden", default=200, type=int)
    parser.add_argument("--rnn_n_layers", default=2, type=int)
    parser.add_argument("--rnn_bptt_len", default=35, type=int)
    parser.add_argument("--rnn_clip", type=float, default=0.25)
    parser.add_argument("--rnn_use_pretrained_emb", type=str2bool, default=True)
    parser.add_argument("--rnn_tie_weights", type=str2bool, default=True)
    parser.add_argument("--rnn_weight_norm", type=str2bool, default=False)

    # miscs
    parser.add_argument("--manual_seed", type=int, default=6, help="manual seed")
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        type=str2bool,
        default=False,
        help="evaluate model on validation set",
    )
    parser.add_argument("--eval_freq", default=1, type=int)
    parser.add_argument("--summary_freq", default=100, type=int)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--track_time", default=False, type=str2bool)
    parser.add_argument("--track_detailed_time", default=False, type=str2bool)
    parser.add_argument("--display_tracked_time", default=False, type=str2bool)
    parser.add_argument("--evaluate_avg", default=False, type=str2bool)

    # checkpoint
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=TRAINING_DIRECTORY,
        type=str,
        help="path to save checkpoint (default: checkpoint)",
    )
    parser.add_argument("--checkpoint_index", type=str, default=None)
    parser.add_argument("--save_all_models", type=str2bool, default=False)
    parser.add_argument("--save_some_models", type=str, default=None)

    """meta info."""
    parser.add_argument("--user", type=str, default="lin")
    parser.add_argument(
        "--project", type=str, default="distributed_adam_type_algorithm"
    )
    parser.add_argument("--experiment", type=str, default=None)

    # device
    parser.add_argument("--backend", type=str, default="mpi")
    parser.add_argument("--use_ipc", type=str2bool, default=False)
    parser.add_argument("--hostfile", type=str, default="iccluster/hostfile")
    parser.add_argument("--mpi_path", type=str, default="$HOME/.openmpi")
    parser.add_argument("--mpi_env", type=str, default=None)
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.6/bin/python"
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--n_mpi_process", default=1, type=int, help="# of the main process."
    )
    parser.add_argument(
        "--n_sub_process",
        default=1,
        type=int,
        help="# of subprocess for each mpi process.",
    )
    parser.add_argument("--world", default=None, type=str)
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    parser.add_argument("--comm_device", type=str, default="cuda")
    parser.add_argument("--local_rank", default=None, type=str)
    parser.add_argument("--clean_python", default=False, type=str2bool)

    # parse conf.
    conf = parser.parse_args()
    if conf.timestamp is None:
        conf.timestamp = get_checkpoint_folder_name(conf)

    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()
