import numpy as np
import configparser as cp
from shutil import copyfile
import os
import argparse

from core.trainer import Trainer
from core.util.experiments import setup_experiment

from core.dataset.preprocessing import ECGDataset
from core.dataset.ecg import BatchGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file")
    args = parser.parse_args()
    configuration_file = args.c
    if os.path.isfile(configuration_file):
        print(f"Using configuration file {configuration_file}")
    else:
        raise FileNotFoundError(f"configuration file {configuration_file} dose not exist")
    np.random.seed(0)
    config = cp.ConfigParser()
    config.read(configuration_file)
    output_dir, tag = setup_experiment(config["DEFAULT"].get("experiments_dir"))
    copyfile(configuration_file, os.path.join(output_dir, configuration_file))
    perct = config["preprocessing"].getfloat("percent_train") / 100
    percv = config["preprocessing"].getfloat("percent_dev") / 100
    mitdb_path = config["mitdb"].get("dataset_path")
    nsrdb_path = config["nsrdb"].get("dataset_path")

    mitdb = ECGDataset.from_directory(mitdb_path, 1)
    nsrdb = ECGDataset.from_directory(nsrdb_path, 0)
    mixture_db = mitdb + nsrdb
    mixture_db.name = "mixture_db"
    print(mitdb, nsrdb, mixture_db, sep="\n")
    training_samples = int(perct * len(mixture_db))
    dev_samples = int(percv * len(mixture_db))
    test_samples = len(mixture_db) - training_samples - dev_samples

    train_set = mixture_db[:training_samples]  # type: ECGDataset
    train_set.name = "training_set"
    dev_set = mixture_db[training_samples:training_samples + dev_samples]  # type: ECGDataset
    dev_set.name = "development_set"
    test_set = mixture_db[training_samples + dev_samples:]  # type: ECGDataset
    test_set.name = "test_set"
    for data_set in [train_set, dev_set, test_set]:
        print(data_set)
        data_set.save(output_dir)
    train_generator = BatchGenerator(train_set, segment_length=config["preprocessing"].getint("sequence_length"),
                                     batch_size=config["preprocessing"].getint("batch_size"))
    dev_generator = BatchGenerator(dev_set, segment_length=config["preprocessing"].getint("sequence_length"),
                                   batch_size=config["preprocessing"].getint("batch_size"))
    RNN_Trainer = Trainer(config, output_dir, tag, logger="trainer")
    RNN_Trainer.train(train_generator, dev_generator)
