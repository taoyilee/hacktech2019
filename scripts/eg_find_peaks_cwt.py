from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    xs = np.arange(0, np.pi, 0.05)
    data = np.sin(xs)
    plt.plot(data)
    plt.show()
    peakind = signal.find_peaks_cwt(data, np.arange(1, 10))
    print(peakind, xs[peakind], data[peakind])
