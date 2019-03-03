from .augmenter import Augmenter

import numpy as np

import random


class RndDCAugmenter(Augmenter):
    def __init__(self, dc = .2, dc_prob = 0.5):
        self.dc = dc

        self.dc_prob = dc_prob

    def augment(self, batch_x: np.ndarray):
        batch_aug = random.choice([batch_x, batch_x+self.dc])

        return batch_aug
