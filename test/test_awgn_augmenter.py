from core.augmenters.awgnaugmenter import AWGNAugmenter
import pytest
import numpy as np


class TestAWGNAugmenter(object):
    @classmethod
    def setup_class(cls):
        cls.awgn_augmenter = AWGNAugmenter(rms_noise_power_percent=0.5)

    @pytest.mark.parametrize("batch_x", [(np.ones((32, 1300, 2))), (np.ones((64, 1300, 2))),
                                         (np.ones((32, 1200, 2))), (np.ones((32, 1300, 4)))])
    def test_output_shape(self, batch_x):
        assert self.awgn_augmenter.augment(batch_x).shape == batch_x.shape

    @pytest.mark.parametrize("rms_noise_power_percent", [10, 5, 15])
    def test_noise_added(self, rms_noise_power_percent):
        augmenter = AWGNAugmenter(rms_noise_power_percent=rms_noise_power_percent)
        batch = np.random.random((32, 1300, 2))
        batch_aug = augmenter.augment(batch)
        assert not np.array_equal(batch, batch_aug)

    @pytest.mark.parametrize("rms_noise_power_percent", [10, 5, 15])
    def test_noise_power(self, rms_noise_power_percent):
        augmenter = AWGNAugmenter(rms_noise_power_percent=rms_noise_power_percent)
        seq_len = 5000
        batch = np.ones((1, seq_len, 2))
        batch = np.concatenate((batch, 2 * np.ones((1, seq_len, 2))), axis=0)
        batch_aug = augmenter.augment(batch)
        diff = np.sqrt(np.mean(np.mean((batch_aug - batch) ** 2, axis=1), axis=1)) / np.array([1, 2])

        assert np.mean(diff) == pytest.approx(rms_noise_power_percent / 100, rel=1e-2)
