# -*- coding: utf-8 -*-
import os

import spacy
from spacy.symbols import ORTH
import torchtext
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from pcode.datasets.loader.imagenet_folder import define_imagenet_folder
from pcode.datasets.loader.svhn_folder import define_svhn_folder
from pcode.datasets.loader.epsilon_or_rcv1_folder import define_epsilon_or_rcv1_folder


"""the entry for classification tasks."""


def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_mnist(root, split, transform, target_transform, download):
    is_train = split == "train"

    if is_train:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    return datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_stl10(root, split, transform, target_transform, download):
    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_svhn(root, split, transform, target_transform, download):
    is_train = split == "train"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return define_svhn_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_imagenet(conf, name, datasets_path, split):
    is_train = split == "train"
    root = (
        os.path.join(
            datasets_path, "lmdb" if "downsampled" not in name else "lmdb_32x32"
        )
        if conf.use_lmdb_data
        else datasets_path
    )

    if is_train:
        root = os.path.join(
            root, "train{}".format("" if not conf.use_lmdb_data else ".lmdb")
        )
    else:
        root = os.path.join(
            root, "val{}".format("" if not conf.use_lmdb_data else ".lmdb")
        )
    return define_imagenet_folder(
        name=name, root=root, flag=conf.use_lmdb_data, cuda=conf.graph.on_cuda
    )


def _get_epsilon_or_rcv1(root, name, split):
    root = os.path.join(root, "{}_{}.lmdb".format(name, split))
    return define_epsilon_or_rcv1_folder(root)


"""the entry for language modeling task."""


def _get_text(batch_first):
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(lower=True, tokenize=spacy_tok, batch_first=batch_first)
    return TEXT


def _get_nlp_lm_dataset(name, datasets_path, batch_first):
    TEXT = _get_text(batch_first)

    # Load and split data.
    if "wikitext2" in name:
        train, valid, test = torchtext.datasets.WikiText2.splits(
            TEXT, root=datasets_path
        )
    elif "ptb" in name:
        train, valid, test = torchtext.datasets.PennTreebank.splits(
            TEXT, root=datasets_path
        )
    else:
        raise NotImplementedError
    return TEXT, train, valid, test


"""the entry for different supported dataset."""


def get_dataset(
    conf,
    name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)

    if name == "cifar10" or name == "cifar100":
        return _get_cifar(name, root, split, transform, target_transform, download)
    elif name == "svhn":
        return _get_svhn(root, split, transform, target_transform, download)
    elif name == "mnist":
        return _get_mnist(root, split, transform, target_transform, download)
    elif name == "stl10":
        return _get_stl10(root, split, transform, target_transform, download)
    elif "imagenet" in name:
        return _get_imagenet(conf, name, datasets_path, split)
    elif name == "epsilon":
        return _get_epsilon_or_rcv1(root, name, split)
    elif name == "rcv1":
        return _get_epsilon_or_rcv1(root, name, split)
    elif name == "wikitext2" or name == "ptb":
        return _get_nlp_lm_dataset(name, datasets_path, batch_first=False)
    else:
        raise NotImplementedError
