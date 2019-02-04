import numpy as np
import matplotlib.pyplot as plt
from core.dataset import LtstdbHea
from scipy import signal

if __name__ == "__main__":
    hea = LtstdbHea.from_file("data/hea/s20271.hea")
    signal0 = hea.signals[0]
    samples = 1000
    signal0 = signal0[:samples]
    # plt.figure()
    # for i in range(1, 34, 1):
    #     plt.plot(signal.daub(i))
    # plt.show()
    plt.figure(figsize=(8, 5))
    wmin = 1
    wmax = 32
    widths = np.arange(wmin, wmax, 2)
    peaks = signal.find_peaks_cwt(signal0, widths, wavelet=lambda x, y: signal.ricker(x, y))
    print(peaks)
    # cwtmatr = signal.cwt(signal0, signal.ricker, widths)
    # plt.imshow(cwtmatr, extent=[0, hea.timestamps[samples], wmin, wmax], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(),
    #            vmin=-abs(cwtmatr).max())
    peak_timestamp = hea.timestamps[peaks]
    # plt.ylim(wmin, wmax)
    plt.vlines(peak_timestamp, wmin, wmax, color="blue")
    scale_factor = (max(signal0) - min(signal0)) / ((wmax - wmin) / 2)
    plt.plot(hea.timestamps[:samples], signal0 / scale_factor + (wmax - wmin) / 2 + wmin, linewidth=2, color="red")
    plt.show()
