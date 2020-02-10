# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch.nn as nn

__all__ = ["lenet"]


class LeNet(nn.Module):
    """
    Input - 3x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, dataset="cifar10"):
        super(LeNet, self).__init__()

        # some init.
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()

        # init layers.
        self.convnet = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(self._decide_input_dim(), 6, kernel_size=(5, 5)),
                    ),
                    ("relu1", nn.ReLU()),
                    ("s2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("conv3", nn.Conv2d(6, 16, kernel_size=(5, 5))),
                    ("relu3", nn.ReLU()),
                    ("s4", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("conv5", nn.Conv2d(16, 120, kernel_size=(5, 5))),
                    ("relu5", nn.ReLU()),
                ]
            )
        )

        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc6", nn.Linear(120, 84)),
                    ("relu6", nn.ReLU()),
                    ("fc7", nn.Linear(84, self.num_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.convnet(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out

    def _decide_num_classes(self):
        if (
            self.dataset == "cifar10"
            or self.dataset == "svhn"
            or self.dataset == "mnist"
        ):
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif self.dataset == "imagenet":
            return 1000

    def _decide_input_dim(self):
        if (
            "cifar" in self.dataset
            or self.dataset == "svhn"
            or self.dataset == "imagenet"
        ):
            return 3
        elif "mnist" == self.dataset:
            return 1
        else:
            raise RuntimeError("incorrect input dim.")


def lenet(conf):
    """Constructs a lenet model."""
    return LeNet(dataset=conf.data)
