from core.signal import LtstdbSignal
import pywt
import numpy as np
from scipy import signal


class RPeakFinder:
    def __init__(self, wavelet="mexh", threshold=0.4):
        self.wavelet = pywt.ContinuousWavelet(wavelet)
        self.threshold = threshold

    def cwt(self, raw_signal: LtstdbSignal, width=1):
        cwtmatr, _ = pywt.cwt(raw_signal, [width], self.wavelet)
        return np.squeeze(cwtmatr[0])

    def find_peaks(self, raw_signal: LtstdbSignal):
        cwtmatr, _ = pywt.cwt(raw_signal, [1], self.wavelet)
        cwtmatr = np.squeeze(cwtmatr[0])
        t = self.threshold * max(cwtmatr)
        # peak_idx = np.array([x if x > t else 0 for x in cwtmatr])
        peak_idx = cwtmatr
        peak_idx = signal.argrelmax(peak_idx)[0]
        return [p - 1 for p in peak_idx]
