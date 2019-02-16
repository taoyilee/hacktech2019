import numpy as np
import glob
import configparser as cp
import matplotlib

from multiprocessing import Pool

from core.util.plotting import RecordPlotter

if __name__ == "__main__":
    matplotlib.use('PDF')
    np.random.seed(0)
    config = cp.ConfigParser()
    config.read("config.ini.template")

    db_paths = {"mitdb": config["mitdb"].get("dataset_path"), "nsrdb": config["nsrdb"].get("dataset_path")}
    for ecg_database, ecg_directory in db_paths.items():
        print(f"Using {ecg_database} from {ecg_directory}")
        record_files = glob.glob(ecg_directory + "*.hea")
        signal_length_sec = 10
        with Pool(8) as p:
            p.map(RecordPlotter(signal_length_sec, ecg_database, 10), record_files)
