from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Bidirectional
from keras import optimizers
import os
from core.action import Action
import logging
from core.util.logger import LoggerFactory
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint
from keras.models import model_from_json, Model


class Tester(Action):
    def __init__(self, config, experiment_env, logger=None):
        super(Tester, self).__init__(config, experiment_env)
        self.logger = LoggerFactory(config).get_logger(logger_name=logger)

    def setup_model(self):
        print("*** Thawing model from JSON ***")
        with open(self.experiment_env.model_json, "r") as fptr:
            json_string = fptr.read()
        model = model_from_json(json_string)  # type:Model
        model.load_weights(self.experiment_env.final_weights)
        model.summary()
        return model

    def predict(self, test_set_generator):
        model = self.setup_model()
        yhat = model.predict_generator(test_set_generator, verbose=1)
        return yhat
