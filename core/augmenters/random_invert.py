from .augmenter import Augmenter
import numpy as np
import random


class RndInvertAugmenter(Augmenter):
    def __init__(self, invert_probability=0.5):
        self.invert_probability = invert_probability

    def augment(self, batch_x: np.ndarray):
        batch_aug = random.choice([batch_x, -batch_x])
        return batch_aug

    def __repr__(self):
        return "Random Y-Inversion augmenter (invert_probability = %.2f)" % self.invert_probability
