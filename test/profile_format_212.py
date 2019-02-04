from line_profiler import LineProfiler
from core.dataset import Format212reader

if __name__ == "__main__":
    profiler = LineProfiler()
    profiler.add_function(Format212reader().from_bytestream)
    profiler.add_function(Format212reader().from_bytes)
    profiler_wrapper = profiler(Format212reader().from_bytestream)
    with open("test/test_cases/short.dat", "rb") as f:
        profiler_wrapper(f)
    profiler.print_stats()
