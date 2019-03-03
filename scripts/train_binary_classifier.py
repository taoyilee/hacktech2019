import argparse
import configparser as cp
import os

from core import Preprocessor
from core import SequenceVisualizer, Trainer
from core.util.experiments import ExperimentEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file", default="config.ini")
    args = parser.parse_args()
    configuration_file = args.c
    if os.path.isfile(configuration_file):
        print("Using configuration file", configuration_file)
    else:
        raise FileNotFoundError("configuration file", configuration_file, "does not exist")
    config = cp.ConfigParser()
    config.read(configuration_file)

    if config["RNN-train"].getboolean("resume_training"):
        print("Resume training from", config["RNN-train"].get("resume_json"))
        experiment_env = ExperimentEnv.from_json(config["RNN-train"].get("resume_json"), config)
    else:
        print("Start a new training")
        experiment_env = ExperimentEnv.setup_training(config)

    train_generator, dev_generator = Preprocessor(config, experiment_env).preprocess()

    if config["RNN-train"].getboolean("plot_datasets"):
        sv = SequenceVisualizer(config, experiment_env)
        sv.visualize(train_generator, batch_limit=50, segment_limit=15)
        sv.visualize(dev_generator, batch_limit=2, segment_limit=15)

    RNN_Trainer = Trainer(config, experiment_env, logger="trainer")
    RNN_Trainer.train(train_generator, dev_generator, experiment_env.model_json)
    experiment_env.write_json()
