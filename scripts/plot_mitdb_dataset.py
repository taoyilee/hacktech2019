import glob
import configparser as cp
import os
from multiprocessing import Pool
from core.util.plotting import RecordPNGPlotterDF, RecordRR
import pandas as pd
import numpy as np

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    mitdb_path = config["mitdb"].get("dataset_path")
    # record_files = glob.glob(os.path.join(mitdb_path, "*.hea"))
    os.makedirs("plot", exist_ok=True)
    signal_length_sec = 10
    df = pd.read_excel("plot/pending_review.xlsx")
    df_plotter = RecordPNGPlotterDF(mitdb_path, signal_length_sec, "mitdb")
    with Pool(4) as p:
        p.map(df_plotter, df.iterrows())
