import wfdb
import configparser as cp
import glob
import os
import numpy as np

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    mitdb_path, nsrdb_path = config["mitdb"].get("dataset_path"), config["nsrdb"].get("dataset_path")
    hea = glob.glob(os.path.join(mitdb_path, "*.hea"))
    for h in hea:
        hea_record = os.path.splitext(h)[0]
        hea_base = os.path.split(hea_record)[1]
        record = wfdb.rdrecord(hea_record)
        signal = record.p_signal
        np.save(os.path.join(mitdb_path, hea_base + ".npy"), signal)

    hea = glob.glob(os.path.join(nsrdb_path, "*.hea"))
    for h in hea:
        hea_record = os.path.splitext(h)[0]
        hea_base = os.path.split(hea_record)[1]
        record = wfdb.rdrecord(hea_record)
        signal = record.p_signal
        np.save(os.path.join(nsrdb_path, hea_base + ".npy"), signal)
