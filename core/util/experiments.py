import os
from datetime import datetime
import numpy as np
import configparser
import json
import random


class ExperimentEnv:
    runtime_parameters = {"tag": "", "experiment_dir": ""}

    def __init__(self, config: configparser.ConfigParser):
        self.config = config
        self.runtime_parameters["experiment_dir"] = config["DEFAULT"].get("experiments_dir")

    def add_key(self, **kwargs):
        self.runtime_parameters.update(kwargs)

    @classmethod
    def setup_training(cls, config: configparser.ConfigParser):
        new_instance = cls(config)
        new_instance.setup_experiment()
        np.random.seed(0)
        with open(os.path.join(new_instance.output_dir, "config.ini"), 'w') as configfile:
            new_instance.config.write(configfile)
        new_instance.write_json()
        return new_instance

    @classmethod
    def setup_testing(cls, config):
        exp_env_json = os.path.join(config["DEFAULT"].get("experiments_dir"),
                                    config["RNN-test"].get("experiment_env_tag"), "experiment_env.json")
        return cls.from_json(exp_env_json, config)

    @classmethod
    def from_json(cls, json_path, config):
        new_instance = cls(config)
        with open(json_path, "r") as fptr:
            new_instance.runtime_parameters = json.load(fptr)
        print("Restoring runtime parameters:")
        for k, v in new_instance.runtime_parameters.items():
            print(k, ":", v)
        return new_instance

    def write_json(self, output_directory=None):
        if output_directory is None:
            output_file = os.path.join(self.runtime_parameters["experiment_dir"], self.runtime_parameters["tag"],
                                       "experiment_env.json")
        else:
            output_file = os.path.join(output_directory, self.runtime_parameters["tag"], "experiment_env.json")
        with open(output_file, "w") as fptr:
            json.dump(self.runtime_parameters, fptr)

    def tag2output_dir(self, tag):
        return os.path.join(self.runtime_parameters["experiment_dir"], tag)

    def setup_experiment(self):
        time_now = datetime.now()
        self.runtime_parameters["tag"] = str(random.getrandbits(40)) + "_" + time_now.strftime("%m%d_%I%M%S")
        print("Experiment tag is", self.runtime_parameters["tag"])
        try:
            os.makedirs(self.tag2output_dir(self.runtime_parameters["tag"]))
        except FileExistsError:
            pass

    @property
    def final_weights(self):
        return self.runtime_parameters["final_weights"]

    @property
    def model_json(self):
        return self.runtime_parameters["model_json"]

    @property
    def test_set(self):
        return self.runtime_parameters["test_set"]

    @property
    def training_set(self):
        return self.runtime_parameters["training_set"]

    @property
    def development_set(self):
        return self.runtime_parameters["development_set"]


    @property
    def tag(self):
        return self.runtime_parameters["tag"]

    @property
    def output_dir(self):
        return self.tag2output_dir(self.runtime_parameters["tag"])
