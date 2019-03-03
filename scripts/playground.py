import configparser as cp
import os
import argparse

from core.util.experiments import ExperimentEnv
from core.dataset import ECGDataset
from core import Preprocessor
from core import SequenceVisualizer

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
    experiment_env = ExperimentEnv.setup_training(config)
    preprocessor = Preprocessor(config, experiment_env)
    # print(preprocessor.mitdb.record_len)
    print(preprocessor.mitdb.tickets[0].get_label(0, 1000))
    print(preprocessor.mitdb.tickets[0].get_label(21606, 30206))

    print(preprocessor.mitdb.tickets[1].get_label(43225, 56000))
    # print(preprocessor.nsrdb.record_len)
    # for t in preprocessor.mitdb.tickets:
    #     print(t)
    # for t in preprocessor.nsrdb.tickets:
    #     print(t)

    # nsrdb = ECGDataset.from_directory(nsrdb_path, config["preprocessing"].getint("NSR_DB_TAG"))
    # train_generator, dev_generator = Preprocessor(config, experiment_env).preprocess(mitdb, nsrdb)
    # sv = SequenceVisualizer(config, experiment_env)
    # sv.visualize(train_generator, batch_limit=None, segment_limit=15)
    # sv.visualize(dev_generator, batch_limit=None, segment_limit=15)
