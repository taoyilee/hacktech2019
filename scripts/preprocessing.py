import configparser as cp
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
        print(f"Using configuration file {configuration_file}")
    else:
        raise FileNotFoundError(f"configuration file {configuration_file} dose not exist")
    config = cp.ConfigParser()
    config.read(configuration_file)
    experiment_env = ExperimentEnv.setup_training(config)
    train_generator, dev_generator = Preprocessor(config, experiment_env).preprocess()

    print(train_generator.dump_labels())
    # print(dev_generator.dump_labels())
    # sv = SequenceVisualizer(config, experiment_env)
    # sv.visualize(train_generator, batch_limit=None, segment_limit=15)
    # sv.visualize(dev_generator, batch_limit=None, segment_limit=15)
