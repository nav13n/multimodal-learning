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
    def __init__(self, trfms):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
            ]
        )
        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(
                    size=32, padding=int(32 * 0.125), padding_mode="reflect"
                ),
                RandAugmentMC(n=2, m=10),
            ]
        )
        self.normalize = trfms

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


# TODO Idetify weak and strong transforms for text
class FixMatchTextTransform(object):
    def __init__(self, trfms):
        self.weak = None
        self.strong = None

    def __call__(self, x):
        weak = x
        strong = x
        return weak, strong
