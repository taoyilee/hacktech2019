from core.signal import LtstdbSignal
import matplotlib.pyplot as plt
import numpy as np


class Interval:
    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __repr__(self):
        return f"({self.start_idx}, {self.end_idx})"

    def plot_signal(self, signal: LtstdbSignal, sampling_period, align=False, shift_idx=0):
        time = sampling_period * np.arange(self.start_idx + shift_idx, self.end_idx + shift_idx)
        if align:
            time -= min(time) - sampling_period * shift_idx
        try:
            plt.plot(time, signal[self.start_idx + shift_idx:self.end_idx + shift_idx])
        except ValueError:
            pass
