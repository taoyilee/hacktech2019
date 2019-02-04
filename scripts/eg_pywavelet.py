import numpy as np
import matplotlib.pyplot as plt
from core.dataset import LtstdbHea
import pywt

if __name__ == "__main__":
    hea = LtstdbHea.from_file("data/hea/s20271.hea")
    signal = hea.signals[0]
    samples = len(signal) // 30000
    print(pywt.families())
    print(pywt.wavelist(kind='discrete'))
    print(f"Samples = {samples}")
    signal = signal[:samples]
    time = hea.timestamps[:samples]
    wavelet = pywt.Wavelet('db1')
    print(wavelet.dec_len, wavelet.dec_hi, wavelet.dec_lo, wavelet.rec_hi, wavelet.rec_lo)
    dec = pywt.wavedec(signal, wavelet, level=6)
    plt.subplot(2, 1, 1)
    plt.plot(time, signal, color="red", linewidth=1, linestyle="--")
    plt.plot(np.linspace(0, time[-1], len(dec[0])), dec[0])
    plt.plot(np.linspace(0, time[-1], len(dec[1])), dec[1])
    plt.plot(np.linspace(0, time[-1], len(dec[2])), dec[2])
    plt.plot(np.linspace(0, time[-1], len(dec[3])), dec[3])
    plt.plot(np.linspace(0, time[-1], len(dec[4])), dec[4])
    plt.plot(np.linspace(0, time[-1], len(dec[5])), dec[5])
    plt.plot(np.linspace(0, time[-1], len(dec[6])), dec[6])
    plt.subplot(2, 1, 2)
    reconstruct0 = pywt.waverec(
        [np.zeros_like(dec[0]), dec[1], dec[2], dec[3], dec[4], np.zeros_like(dec[5]), np.zeros_like(dec[6])],
        wavelet, 'smooth')
    reconstruct1 = pywt.waverec(
        [dec[0], np.zeros_like(dec[1]), dec[2], dec[3], dec[4], dec[5], np.zeros_like(dec[6])],
        wavelet, 'smooth')
    # plt.plot(np.linspace(0, time[-1], len(reconstruct0)), reconstruct0)
    plt.plot(np.linspace(0, time[-1], len(reconstruct1)), reconstruct1)
    plt.plot(time, signal, color="red", linewidth=1, linestyle="--")
    plt.show()
