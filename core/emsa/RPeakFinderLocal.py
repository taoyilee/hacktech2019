from core.signal import LtstdbSignal
import numpy as np
from scipy import signal


class RPeakFinderLocal:
    def __init__(self, search=1):
        self.search = search

    def tweak_peak_index(self, raw_signal: LtstdbSignal, peak_idx):
        n = len(raw_signal)

        def find_near_peak_idx(p):
            idx = signal.argrelmax(raw_signal[p - self.search:p + self.search + 1], mode="wrap")[0]
            if len(idx) == 0:
                return None
            else:
                idx = idx[0]
            idx -= self.search
            return min(n, max(0, p + idx))

        raw_peaks = map(find_near_peak_idx, peak_idx)
        raw_peaks = list(filter(lambda x: x is not None, raw_peaks))
        return np.unique(raw_peaks).tolist()
