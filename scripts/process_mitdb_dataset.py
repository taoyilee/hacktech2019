import glob
import configparser as cp
import os
from multiprocessing import Pool
from core.util.plotting import RecordPNGPlotter, RecordRR
import pandas as pd
import numpy as np

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    mitdb_path = config["mitdb"].get("dataset_path")
    record_files = glob.glob(os.path.join(mitdb_path, "*.hea"))
    os.makedirs("plot", exist_ok=True)
    signal_length_sec = 10
    with Pool(4) as p:
        data = p.map(RecordRR(signal_length_sec, "mitdb"), record_files)

    data = np.array(data, dtype=object).squeeze()
    print(data.shape)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    df = pd.DataFrame(data=data,
                      columns=["Record", "Start_Index", "End_Index", "Arrhythmia", "Max_RRi", "Min_RRi", "Max_HRV_perc",
                               "Min_HRV_perc", "hrv_perc", "RR Interval"])
    df.to_excel("plot/mitdb_summary.xlsx")
