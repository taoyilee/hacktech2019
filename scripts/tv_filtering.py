from core.dataset import LtstdbHea
from core.filtering.tv_filter import TvFilter
from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from scipy.sparse import diags

if __name__ == "__main__":
    plt.close("all")
    hea = LtstdbHea.from_file("data/hea/s20271.hea")
    samples = 500
    resampled_samples = samples
    signal0 = hea.signals[0]
    signal = resample(signal0[:samples], resampled_samples)
    time_resample = np.linspace(0, hea.timestamps[samples], resampled_samples)
    # delta = np.linspace(0.01, 1, 10)
    delta = [0.1, 0.5]
    delta = np.round(delta, 3)
    results = np.empty_like(delta, dtype=OptimizeResult)
    for i, d in enumerate(delta):
        flt = TvFilter(delta=d, method='BFGS')
        results[i] = flt.filter(signal)  # type: OptimizeResult
    plt.figure(figsize=(10, 14))
    plt.subplot(len(delta) + 1, 1, 1)
    plt.plot(time_resample, signal, color="red", linestyle="-", linewidth=2)
    plt.plot(hea.timestamps[:samples], signal0[:samples], color="red", linestyle="--", linewidth=1)
    plt.grid()
    for i, r in enumerate(results):
        plt.subplot(len(delta) + 1, 1, i + 2)
        plt.plot(time_resample, r.x)
        plt.title(f"delta={delta[i]}")
        plt.grid()
    plt.show()

    D = diags([-1, 1], offsets=[0, 1], shape=(resampled_samples - 1, resampled_samples))
    plt.figure()
    for i, r in enumerate(results):
        xhat = r.x
        Dxhat = D * xhat
        phi_tv = np.linalg.norm(Dxhat, ord=1) / resampled_samples
        error = np.linalg.norm(xhat - signal, ord=2) / resampled_samples
        plt.scatter(error, phi_tv, label=f"delta={delta[i]}")
    plt.legend()
    plt.grid()
    plt.show()
