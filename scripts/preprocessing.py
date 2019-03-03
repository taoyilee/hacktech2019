import configparser as cp
import numpy as np
import os
import argparse
from core.util.experiments import ExperimentEnv
from core import Preprocessor
from core import SequenceVisualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file")
    args = parser.parse_args()
    configuration_file = args.c
    if os.path.isfile(configuration_file):
        print("Using configuration file", configuration_file)
    else:
        raise FileNotFoundError("configuration file", configuration_file, "does not exist")
    config = cp.ConfigParser()
    config.read(configuration_file)
    experiment_env = ExperimentEnv.setup_training(config)
    train_generator, dev_generator = Preprocessor(config, experiment_env).preprocess()

    for b in train_generator:
        print(b[0].shape, b[1].shape)

    train_labels = train_generator.dump_labels()
    dev_labels = dev_generator.dump_labels()

    np.save("Train_labels.npy", train_labels)
    np.save("Dev_labels.npy", dev_labels)
    unique, counts = np.unique(train_generator.dump_labels(), return_counts=True)
    print(unique)
    print(counts)
    print()
    unique, counts = np.unique(dev_generator.dump_labels(), return_counts=True)
    print(unique)
    print(counts)
    print()

    """sv = SequenceVisualizer(config, experiment_env)
    sv.visualize(train_generator, batch_limit=None, segment_limit=15)
    sv.visualize(dev_generator, batch_limit=None, segment_limit=15)"""
