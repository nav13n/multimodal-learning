import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

# Adapted from sources:
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/cifar.py
class FixMatchImageTransform(object):
    def __init__(self, pre_trfms, post_trfms):
        self.weak = transforms.Compose(
            [
                transforms.Resize(size=(300, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=224, padding=int(32 * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.Resize(size=(300, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=224, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.pre = pre_trfms
        self.post = post_trfms

    def __call__(self, x):
        x = self.pre(x)
        weak = self.weak(x)
        strong = self.strong(x)
        return self.post(weak), self.post(strong)


# TODO Idetify weak and strong transforms for text
class FixMatchTextTransform(object):
    def __init__(self, trfms):
        self.weak = None
        self.strong = None

    def __call__(self, x):
        weak = x
        strong = x
        return weak, strong
