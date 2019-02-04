from core.dataset.ltstdb_hea import LtstdbHea
import os
from core.util.plotting import plot_hea, plot_hea_biosppy

if __name__ == "__main__":
    for root, dirs, files in os.walk(os.path.join("data", "hea")):
        for hea_file in files:
            if hea_file[-4:] == ".hea":
                print(os.path.join(root, hea_file))
                hea = LtstdbHea.from_file(os.path.join(root, hea_file))
                plot_hea(hea, samples=10000)
                plot_hea_biosppy(hea, samples=10000)
