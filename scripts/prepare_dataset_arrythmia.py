import numpy as np
import glob
from core.dataset.physio_bank import Dataset
import configparser as cp
from shutil import copyfile
import os
import logging
from core.dataset.qtdb import load_dat, split_dataset
from core.models.rnn import get_model
from core.util.experiments import setup_experiment

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
config = cp.ConfigParser()
config.read("config.ini")
try:
    os.makedirs(config["logging"].get("logdir"))
except FileExistsError:
    pass
fh = logging.FileHandler(os.path.join(config["logging"].get("logdir"), "main.log"), mode="w+")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":
    configuration_file = "config.ini"
    np.random.seed(0)
    config = cp.ConfigParser()
    config.read(configuration_file)
    nsrdb_path = config["nsrdb"].get("dataset_path")
    print(f"Using NSR dataset from {nsrdb_path}")
    mitdb_path = config["mitdb"].get("dataset_path")
    print(f"Using MIT dataset from {mitdb_path}")

    dataset_out = config["RNN-arrhythmia"].get("dataset_out")
    training_percent = config["RNN-arrhythmia"].get("training_percent")
    validation_percent = config["RNN-arrhythmia"].get("validation_percent")
    testing_percent = config["RNN-arrhythmia"].get("testing_percent")

    nsrdb = Dataset.from_dir(nsrdb_path)
    mitdb = Dataset.from_dir(mitdb_path)

    mitnsrdb = nsrdb + mitdb
