from core.util.experiments import ExperimentEnv
import configparser


class Action:
    def __init__(self, config: configparser.ConfigParser, experiment_env: ExperimentEnv):
        self.experiment_env = experiment_env
        self.config = config
