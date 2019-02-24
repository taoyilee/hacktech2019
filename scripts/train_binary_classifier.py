import configparser as cp
from core import Preprocessor
import os
import argparse
from core.util.experiments import ExperimentEnv
from core.dataset.preprocessing import ECGDataset
from core import SequenceVisualizer, Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file")
    args = parser.parse_args()
    configuration_file = args.c
    if os.path.isfile(configuration_file):
        print(f"Using configuration file {configuration_file}")
    else:
        raise FileNotFoundError(f"configuration file {configuration_file} dose not exist")
    config = cp.ConfigParser()
    config.read(configuration_file)
    experiment_env = ExperimentEnv(config)
    mitdb_path, nsrdb_path = config["mitdb"].get("dataset_path"), config["nsrdb"].get("dataset_path")
    mitdb = ECGDataset.from_directory(mitdb_path, config["preprocessing"].getint("MIT_DB_TAG"))
    nsrdb = ECGDataset.from_directory(nsrdb_path, config["preprocessing"].getint("NSR_DB_TAG"))
    train_generator, dev_generator = Preprocessor(config, experiment_env).preprocess(mitdb, nsrdb)

    if config["RNN-train"].getboolean("plot_datasets"):
        sv = SequenceVisualizer(config, experiment_env)
        sv.visualize(train_generator, batch_limit=None, segment_limit=15)
        sv.visualize(dev_generator, batch_limit=None, segment_limit=15)

    RNN_Trainer = Trainer(config, experiment_env, logger="trainer")
    RNN_Trainer.train(train_generator, dev_generator)
