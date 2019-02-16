import numpy as np
import glob
import os
from core.dataset.qtdb import load_dat
from core.util.experiments import setup_experiment
import configparser as cp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from core.util.plotting import plotecg

if __name__ == "__main__":
    matplotlib.use('PDF')
    np.random.seed(0)
    config = cp.ConfigParser()
    config.read("config.ini.template")
    output_dir, tag = setup_experiment(config["DEFAULT"].get("experiments_dir"))
    REJECTED_TAGS = tuple(config["qtdb"].get("reject_tags").split(","))
    VALID_SEGMTS = tuple(config["qtdb"].get("valid_segments").split(","))
    CATEGORIES = tuple([int(i) for i in config["qtdb"].get("category").split(",")])

    qtdbpath = config["qtdb"].get("dataset_path")
    print(f"Using qtdb dataset from {qtdbpath}")
    perct = config["qtdb"].getfloat("training_percent")
    percv = config["qtdb"].getfloat("validation_percent")

    exclude = set()
    exclude.update(config["qtdb"].get("excluded_records").split(","))

    initial_weights = config["RNN-train"].get("initial_weights")
    model_output = config["RNN-train"].get("model_output")
    epochs = config["RNN-train"].getint("epochs")

    datfiles = glob.glob(qtdbpath + "*.dat")
    tagged_data = load_dat(datfiles, VALID_SEGMTS, CATEGORIES, exclude, REJECTED_TAGS)
    with PdfPages(os.path.join('plot/dataset_view.pdf')) as pdf:
        for data in tagged_data:
            end_index = min(data.y.shape[0], 1000)
            plotecg(data.x, data.y, 0, end_index, fs=data.fs)
            pdf.savefig()
            plt.close()

    with PdfPages(os.path.join('plot/dataset_view_full.pdf')) as pdf:
        for data in tagged_data:
            end_index = data.y.shape[0]
            plotecg(data.x, data.y, 0, end_index, fs=data.fs)

            pdf.savefig()
            plt.close()
