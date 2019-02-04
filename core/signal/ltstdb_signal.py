from core.dataset.format_212_reader import Format212reader
import numpy as np


class LtstdbSignal:
    signal = []

    def __init__(self, signal):
        self.signal = signal

    def __getitem__(self, item):
        return self.signal[item]

    @classmethod
    def from_file(cls, dat_file, num_signals=2, format=212):
        """

        :param dat_file:
        :param num_signals:
        :param format:
        :return:
        """
        reader = Format212reader()
        with open(dat_file, "rb") as f:
            raw_signal = reader.from_bytestream(f)  # type: np.ndarray
        return tuple(cls(raw_signal[i::num_signals]) for i in range(num_signals))

    def __len__(self):
        return len(self.signal)
