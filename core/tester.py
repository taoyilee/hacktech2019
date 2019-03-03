from core.action import Action
from core.util.logger import LoggerFactory
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras import optimizers


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
        adam = optimizers.Adam(lr=self.config["RNN-train"].getfloat("initial_lr"))
        model.compile(loss='binary_crossentropy', optimizer=adam)
        model.summary()
        return model

    def predict(self, test_set_generator):
        model = self.setup_model()
        yhat = model.predict_generator(test_set_generator, verbose=1)
        return yhat
