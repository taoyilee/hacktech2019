import base64
import hashlib
import os
from datetime import datetime
import numpy as np
import configparser


class ExperimentEnv:
    def __init__(self, config: configparser.ConfigParser):
        self._tag = ""
        self.config = config
        self.experiment_dir = config["DEFAULT"].get("experiments_dir")
        self.setup_experiment()
        np.random.seed(0)
        with open(os.path.join(self.output_dir, "config.ini"), 'w') as configfile:
            config.write(configfile)

    def tag2output_dir(self, tag):
        return os.path.join(self.experiment_dir, tag)

    def setup_experiment(self):
        time_now = datetime.now()
        hashfun = hashlib.sha1()
        hashfun.update(b"{time_now}")
        self._tag = base64.b64encode(hashfun.digest()).decode()[:10] + "_" + time_now.strftime("%m%d_%I%M%S")
        print(f"Experiment tag is {self._tag}")
        try:
            os.makedirs(self.tag2output_dir(self._tag))
        except FileExistsError:
            pass

    @property
    def tag(self):
        return self._tag

    @property
    def output_dir(self):
        return self.tag2output_dir(self._tag)
