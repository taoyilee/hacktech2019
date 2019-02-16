import numpy as np
# import glob
import configparser as cp
from shutil import copyfile
import os
import argparse
from core.util.experiments import setup_experiment

from core.dataset.preprocessing import ECGDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file")
    args = parser.parse_args()
    configuration_file = args.c
    print(f"Using configuration file {configuration_file}")
    np.random.seed(0)
    config = cp.ConfigParser()
    config.read(configuration_file)
    output_dir, tag = setup_experiment(config["DEFAULT"].get("experiments_dir"))
    copyfile(configuration_file, os.path.join(output_dir, configuration_file))
    REJECTED_TAGS = tuple(config["qtdb"].get("reject_tags").split(","))
    VALID_SEGMTS = tuple(config["qtdb"].get("valid_segments").split(","))
    CATEGORIES = tuple([int(i) for i in config["qtdb"].get("category").split(",")])

    qtdbpath = config["qtdb"].get("dataset_path")
    print(f"Using qtdb dataset from {qtdbpath}")
    perct = config["preprocessing"].getfloat("percent_train")
    percv = config["preprocessing"].getfloat("percent_dev")
    mitdb_path = config["mitdb"].get("dataset_path")
    nsrdb_path = config["nsrdb"].get("dataset_path")

    # TODO: lazy load the dataset
    mitdb = ECGDataset.from_directory(mitdb_path, 1)
    nsrdb = ECGDataset.from_directory(nsrdb_path, 0)
    mixture_db = mitdb + nsrdb
    mixture_db.name = "mixture_db"
    print(mitdb, nsrdb, mixture_db, sep="\n")

    training_samples = int(perct * len(mixture_db))
    dev_samples = int(percv * len(mixture_db))
    # test_samples = training_samples + dev_samples
    test_samples = len(mixture_db) - training_samples - dev_samples

    train_set = mixture_db[:training_samples]
    train_set.name = "training set"
    dev_set = mixture_db[training_samples:training_samples + dev_samples]
    dev_set.name = "development set"
    test_set = mixture_db[training_samples + dev_samples:]
    test_set.name = "test set"
    print(train_set)
    print(dev_set)
    print(test_set)

    train_generator = train_set.create_generator()
