from line_profiler import LineProfiler
from core.emsa.RPeakFinder import RPeakFinder
from core.dataset import LtstdbHea

if __name__ == "__main__":
    profiler = LineProfiler()
    profiler.add_function(RPeakFinder().find_peaks)

    profiler_wrapper = profiler(RPeakFinder().find_peaks)

    hea = LtstdbHea.from_file("data/hea/s20211.hea")
    raw_signal = hea.signals[0][:1000]
    time = hea.timestamps[:1000]
    peak_finder = RPeakFinder()
    peak_idx = profiler_wrapper(raw_signal)
    profiler.print_stats()
