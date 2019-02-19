from keras.models import Sequential
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, Bidirectional
from keras import optimizers
import os
import logging

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, CSVLogger, ModelCheckpoint


class Trainer:
    logger = logging.getLogger('trainer')

    def __init__(self, config, output_directory, tag):
        self.config = config
        self.output_dir = output_directory
        self.tag = tag
        self.setup_logger()

    def setup_logger(self):
        self.logger.handlers = []
        self.logger.setLevel(logging.DEBUG)
        try:
            os.makedirs(self.config["DEFAULT"].get("logdir"))
        except FileExistsError:
            pass
        fh = logging.FileHandler(os.path.join(self.config["DEFAULT"].get("logdir"), "trainer.log"), mode="w+")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_optimizer(self, model):
        print(f"*** Adding optimizer to the model ***")
        adam = optimizers.adam(lr=self.config["RNN-train"].getfloat("initial_lr"))
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

    def setup_model(self):
        print(f"*** Setting up deep learning model ***")
        input_shape = (self.config["preprocessing"].getint("sequence_length"), 2)
        model = Sequential()
        model.add(Bidirectional(LSTM(self.config["RNN-train"].getint("rnn_output_features")), input_shape=input_shape))
        model.add(Dropout(self.config["RNN-train"].getfloat("dropout")))
        model.add(BatchNormalization())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(self.config["RNN-train"].getfloat("dropout")))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        self.setup_optimizer(model)
        return model

    def setup_callbacks(self):
        print(f"*** Setting up callbacks ***")
        callbacks = [
            ModelCheckpoint(os.path.join(self.output_dir, "weights.{epoch:02d}.h5"), monitor='val_loss', verbose=0,
                            save_best_only=False, save_weights_only=False, mode='auto', period=1),
            CSVLogger(os.path.join(self.output_dir, f"training.csv"), separator=',', append=False),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=self.config["RNN-train"].getint("patientce_reduce_lr"),
                              verbose=self.config["RNN-train"].getint("verbosity"), mode='min',
                              cooldown=0, min_lr=1e-12),
            TensorBoard(log_dir=os.path.join(self.config["RNN-train"].get("tensorboard_dir"), self.tag))]
        if self.config["RNN-train"].getboolean("early_stop"):
            self.logger.log(logging.INFO, f"Early Stop enabled")
            callbacks.append(
                EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=5, verbose=1, mode='min'))
        else:
            self.logger.log(logging.INFO, f"Early Stop disabled")

        return callbacks

    def train(self, training_set_generator, dev_set_generator):
        model = self.setup_model()
        with open(os.path.join(self.output_dir, "model.json"), "w") as f:
            f.write(model.to_json())
        model.fit_generator(generator=training_set_generator, validation_data=dev_set_generator,
                            epochs=self.config["RNN-train"].getint("epochs"), verbose=1,
                            callbacks=self.setup_callbacks())
        model.save(os.path.join(self.output_dir, "final_weights.h5"))
