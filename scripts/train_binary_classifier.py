import configparser as cp
from core import Preprocessor
import os
import argparse
from core.util.experiments import ExperimentEnv
from core import SequenceVisualizer, Trainer

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

    if config["RNN-train"].getboolean("plot_datasets"):
        sv = SequenceVisualizer(config, experiment_env)
        sv.visualize(train_generator, batch_limit=50, segment_limit=15)
        sv.visualize(dev_generator, batch_limit=2, segment_limit=15)

    RNN_Trainer = Trainer(config, experiment_env, logger="trainer")
    RNN_Trainer.train(train_generator, dev_generator)
    experiment_env.write_json()
