from .augmenter import Augmenter
import numpy as np


class AWGNAugmenter(Augmenter):
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio

    def augment(self, batch: np.ndarray):
        return np.array([])
