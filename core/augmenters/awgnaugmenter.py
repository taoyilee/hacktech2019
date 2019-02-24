from .augmenter import Augmenter
import numpy as np


class AWGNAugmenter(Augmenter):
    def __init__(self, rms_noise_power_percent):
        self.rms_noise_power_ratio = rms_noise_power_percent / 100

    def augment(self, batch_x: np.ndarray):
        sequence_axis = 1
        sequence_length = batch_x.shape[sequence_axis]
        batch_x_rms = np.sqrt(np.mean(batch_x ** 2, axis=sequence_axis))
        noise = np.random.normal(0, self.rms_noise_power_ratio * batch_x_rms,
                                 (sequence_length, batch_x_rms.shape[0], batch_x_rms.shape[1])).swapaxes(0, 1)
        batch_aug = batch_x + noise
        return batch_aug

    def __repr__(self):
        return f"AWGN augmenter (rms_noise_power_ratio = {self.rms_noise_power_ratio})"
