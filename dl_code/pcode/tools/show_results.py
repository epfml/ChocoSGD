# -*- coding: utf-8 -*-
from __future__ import division
import os
import json
import functools
import numbers
import pandas as pd


from pcode.utils.op_paths import list_files
from pcode.utils.op_files import load_pickle
from pcode.utils.auxiliary import str2time


"""load data from pickled file."""


def _get_arguments(arguments):
    arguments = vars(arguments)

    arguments.update(
        {
            "n_nodes": arguments["graph"].n_nodes,
            "world": arguments["graph"].world,
            "rank": arguments["graph"].rank,
            "ranks": arguments["graph"].ranks,
            "device": arguments["graph"].device,
            "on_cuda": arguments["graph"].on_cuda,
        }
    )
    arguments["graph"] = None
    return arguments


def get_pickle_info(root_data_path, experiments):
    file_paths = []
    for experiment in experiments:
        file_paths += [
            os.path.join(root_data_path, experiment, file)
            for file in os.listdir(os.path.join(root_data_path, experiment))
            if "pickle" in file
        ]

    results = dict((path, load_pickle(path)) for path in file_paths)
    info = functools.reduce(lambda a, b: a + b, list(results.values()))
    return info


"""load the raw results"""


def load_raw_info_from_experiments(root_path):
    """load experiments.
        root_path: a directory with a list of different trials.
    """
    exp_folder_paths = [
        folder_path
        for folder_path in list_files(root_path)
        if "pickle" not in folder_path
    ]

    info = []
    for folder_path in exp_folder_paths:
        try:
            element_of_info = _get_info_from_the_folder(folder_path)
            info.append(element_of_info)
        except Exception as e:
            print("error: {}".format(e))
    return info


def _get_info_from_the_folder(folder_path):
    print("process the folder: {}".format(folder_path))
    arguments_path = os.path.join(folder_path, "arguments.pickle")

    # collect runtime json info for one rank.
    sub_folder_paths = sorted(
        [
            sub_folder_path
            for sub_folder_path in list_files(folder_path)
            if ".tar" not in sub_folder_path and "pickle" not in sub_folder_path
        ]
    )

    # return the information.
    return (
        folder_path,
        {
            "arguments": _get_arguments(load_pickle(arguments_path)),
            # single worker records.
            "single_records": _parse_runtime_infos(os.path.join(sub_folder_paths[0])),
        },
    )


def _parse_runtime_infos(file_folder):
    existing_json_files = [file for file in os.listdir(file_folder) if "json" in file]

    if "log.json" in existing_json_files:
        # old logging scheme.
        return _parse_runtime_info(os.path.join(file_folder, "log.json"))
    else:
        # new logging scheme.
        lines = []
        for idx in range(1, 1 + len(existing_json_files)):
            _lines = _parse_runtime_info(
                os.path.join(file_folder, "log-{}.json".format(idx))
            )
            lines.append(_lines)

        return functools.reduce(
            lambda a, b: [a[idx] + b[idx] for idx in range(len(a))], lines
        )


def _parse_runtime_info(json_file_path):
    with open(json_file_path) as json_file:
        lines = json.load(json_file)

        # distinguish lines to different types.
        tr_lines, te_lines, te_avg_lines = [], [], []

        for line in lines:
            if line["measurement"] != "runtime":
                continue

            try:
                _time = str2time(line["time"], "%Y-%m-%d %H:%M:%S")
            except:
                _time = None
            line["time"] = _time

            if line["split"] == "train":
                tr_lines.append(line)
            elif line["split"] == "test":
                if line["type"] == "local_model":
                    te_lines.append(line)
                elif line["type"] == "local_model_avg":
                    te_avg_lines.append(line)
    return tr_lines, te_lines, te_avg_lines


"""extract the results based on the condition."""


def _is_same(items):
    return len(set(items)) == 1


def is_meet_conditions(args, conditions, threshold=1e-8):
    if conditions is None:
        return True

    # get condition values and have a safety check.
    condition_names = list(conditions.keys())
    condition_values = list(conditions.values())
    assert _is_same([len(values) for values in condition_values]) is True

    # re-build conditions.
    num_condition = len(condition_values)
    num_condition_value = len(condition_values[0])
    condition_values = [
        [condition_values[ind_cond][ind_value] for ind_cond in range(num_condition)]
        for ind_value in range(num_condition_value)
    ]

    # check re-built condition.
    g_flag = False
    try:
        for cond_values in condition_values:
            l_flag = True
            for ind, cond_value in enumerate(cond_values):
                _cond = cond_value == args[condition_names[ind]]

                if isinstance(cond_value, numbers.Number):
                    _cond = (
                        _cond
                        or abs(cond_value - args[condition_names[ind]]) <= threshold
                    )

                l_flag = l_flag and _cond
            g_flag = g_flag or l_flag
        return g_flag
    except:
        return False


def reorganize_records(records):
    def _parse(lines, is_train=True):
        time, step, loss, top1, top5, ppl, bits = [], [], [], [], [], [], []

        for line in lines:
            time.append(line["time"])
            step.append(line["local_index"] if is_train else line["epoch"])
            loss.append(line["loss"])
            top1.append(line["top1"] if "top1" in line else 0)
            top5.append(line["top5"] if "top5" in line else 0)
            ppl.append(line["ppl"] if "ppl" in line else 0)
            bits.append(line["n_bits_to_transmit"] if is_train else 0)
        return time, step, loss, top1, top5, ppl, bits

    # deal with single records.
    tr_records, te_records, te_avg_records = records["single_records"]
    te_records = [record for record in te_records]
    tr_time, tr_step, tr_loss, tr_top1, tr_top5, tr_ppl, tr_MB = _parse(
        tr_records, is_train=True
    )
    te_time, te_epoch, te_loss, te_top1, te_top5, te_ppl, _ = _parse(
        te_records, is_train=False
    )
    te_avg_perf = [0] + [record["best_perf"] for record in te_avg_records]

    # return all results.
    return {
        "tr_time": tr_time,
        "tr_MB": tr_MB,
        "tr_loss": tr_loss,
        "tr_top1": tr_top1,
        "tr_top5": tr_top5,
        "tr_ppl": tr_ppl,
        "te_time": te_time,
        "te_step": te_epoch,
        "te_loss": te_loss,
        "te_top1": te_top1,
        "te_top1_upon": [max(te_top1[:idx]) for idx in range(1, 1 + len(te_top1))],
        "te_top5": te_top5,
        "te_top5_upon": [max(te_top5[:idx]) for idx in range(1, 1 + len(te_top5))],
        "te_ppl": te_ppl,
        "te_ppl_upon": [max(te_ppl[:idx]) for idx in range(1, 1 + len(te_ppl))],
        "te_avg_perf": te_avg_perf,
        "te_avg_perf_upon": [
            max(te_avg_perf[:idx]) for idx in range(1, 1 + len(te_avg_perf))
        ],
    }


def extract_list_of_records(list_of_records, conditions):
    # load and filter data.
    records = []

    for path, raw_records in list_of_records:
        # check conditions.
        if not is_meet_conditions(raw_records["arguments"], conditions):
            continue

        # get parsed records
        records += [(raw_records["arguments"], reorganize_records(raw_records))]

    print("we have {}/{} records.".format(len(records), len(list_of_records)))
    return records


"""summary the results."""


def _summarize_info(record, arg_names, be_groupby, larger_is_better):
    args, info = record
    test_performance = (
        max(info[be_groupby]) if larger_is_better else min(info[be_groupby])
    )
    return [args[arg_name] if arg_name in args else None for arg_name in arg_names] + [
        test_performance
    ]


def reorder_records(records, based_on):
    # records is in the form of <args, info>
    conditions = based_on.split(",")
    list_of_args = [
        (ind, [args[condition] for condition in conditions])
        for ind, (args, info) in enumerate(records)
    ]
    sorted_list_of_args = sorted(list_of_args, key=lambda x: x[1:])
    return [records[ind] for ind, args in sorted_list_of_args]


def summarize_info(records, arg_names, be_groupby="te_top1", larger_is_better=True):
    # define header.
    headers = arg_names + [be_groupby]
    # reorder records
    records = reorder_records(records, based_on="n_nodes")
    # extract test records
    test_records = [
        _summarize_info(record, arg_names, be_groupby, larger_is_better)
        for record in records
    ]
    # aggregate test records
    aggregated_records = pd.DataFrame(test_records, columns=headers)
    # average test records
    averaged_records = (
        aggregated_records.fillna(-1)
        .groupby(headers[:-1], as_index=False)
        .agg({be_groupby: ["mean", "std", "max", "min", "count"]})
        .sort_values((be_groupby, "mean"), ascending=not larger_is_better)
    )
    return aggregated_records, averaged_records
