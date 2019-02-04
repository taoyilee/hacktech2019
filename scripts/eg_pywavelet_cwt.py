import matplotlib.pyplot as plt
from core.dataset import LtstdbHea
from core.emsa.RPeakFinder import RPeakFinder
from core.emsa import RPeakFinderLocal
from core.emsa.Interval import Interval
import os
from scipy import signal

if __name__ == "__main__":
    os.makedirs(os.path.join("plot", "segment"), exist_ok=True)
    for root, dirs, files in os.walk(os.path.join("data", "hea")):
        for hea_file in files:
            if hea_file[-4:] == ".hea":
                hea_file = os.path.join(root, hea_file)
                print(hea_file)
                hea = LtstdbHea.from_file(hea_file)
                samples = len(hea.signals[0]) // 10000
                raw_signal = hea.signals[0][:samples]
                time = hea.timestamps[:samples]
                peak_finder = RPeakFinder(threshold=0.2)
                peak_finder_tweak = RPeakFinderLocal(search=2)

                peak_idx = peak_finder.find_peaks(raw_signal)
                peak_idx = peak_finder_tweak.tweak_peak_index(raw_signal, peak_idx)
                print(peak_idx)
                plt.figure()
                plt.plot(time, raw_signal)
                cwt = peak_finder.cwt(raw_signal, width=1)
                plt.plot(time, cwt)
                plt.scatter(time[peak_idx], raw_signal[peak_idx], color="red", s=5)
                plt.savefig(os.path.join("plot", "segment", f"{hea.name}_peak.png"))
                plt.figure()
                plt.hist(cwt[signal.argrelmax(cwt)], bins=20, density=True)
                plt.savefig(os.path.join("plot", "segment", f"{hea.name}_hist.png"))

                intervals = []
                for p, pp in zip(peak_idx[:-1], peak_idx[1:]):
                    intervals.append(Interval(p, pp))
                plt.figure()
                for intv in intervals:
                    intv.plot_signal(raw_signal, hea.sampling_period, align=True, shift_idx=-80)
                plt.grid()
                plt.savefig(os.path.join("plot", "segment", f"{hea.name}_seg.png"))
            plt.close("all")
