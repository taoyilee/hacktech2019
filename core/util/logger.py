import os
import logging


class LoggerFactory:

    def __init__(self, config):
        self.config = config

    @classmethod
    def dummy(cls):
        return logging.getLogger('dummy')

    def get_logger(self, logger_name=None, logging_level=logging.DEBUG):
        if logger_name is None:
            return logging.getLogger('dummy')
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.setLevel(logging_level)
        try:
            os.makedirs(self.config["DEFAULT"].get("logdir"))
        except FileExistsError:
            pass
        fh = logging.FileHandler(os.path.join(self.config["DEFAULT"].get("logdir"), f"{logger_name}.log"), mode="w+")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger
