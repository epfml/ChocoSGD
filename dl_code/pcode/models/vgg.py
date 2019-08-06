# -*- coding: utf-8 -*-
import math

import torch.nn as nn


__all__ = ['vgg']


ARCHITECTURES = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, nn_arch, dataset, use_bn=True):
        super(VGG, self).__init__()

        # init parameters.
        self.use_bn = use_bn
        self.nn_arch = nn_arch
        self.dataset = dataset
        self.num_classes = self._decide_num_classes()

        # init models.
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes),
        )

        # weight initialization.
        self._weight_initialization()

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _decide_num_classes(self):
        if self.dataset == 'cifar10' or self.dataset == 'svhn':
            return 10
        elif self.dataset == 'cifar100':
            return 100
        else:
            raise ValueError('not allowed dataset.')

    def _make_layers(self):
        layers = []
        in_channels = 3
        for v in ARCHITECTURES[self.nn_arch]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.use_bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def vgg(conf):
    use_bn = 'bn' in conf.arch
    dataset = conf.data

    if '11' in conf.arch:
        return VGG(nn_arch='A', dataset=dataset, use_bn=use_bn)
    elif '13' in conf.arch:
        return VGG(nn_arch='B', dataset=dataset, use_bn=use_bn)
    elif '16' in conf.arch:
        return VGG(nn_arch='D', dataset=dataset, use_bn=use_bn)
    elif '19' in conf.arch:
        return VGG(nn_arch='E', dataset=dataset, use_bn=use_bn)
    else:
        raise NotImplementedError
