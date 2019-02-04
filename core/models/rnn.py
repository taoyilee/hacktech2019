import numpy as np
import os
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.layers import LSTM, Bidirectional
from keras.models import Sequential
from keras import optimizers, regularizers
from keras.layers.normalization import BatchNormalization
import logging
import configparser as cp

np.random.seed(0)
logger = logging.getLogger('rnn')
logger.setLevel(logging.DEBUG)
config = cp.ConfigParser()
config.read("config.ini")
try:
    os.makedirs(config["logging"].get("logdir"))
except FileExistsError:
    pass
fh = logging.FileHandler(os.path.join(config["logging"].get("logdir"), "rnn.log"), mode="w+")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def get_session(gpu_fraction=0.8):
    # allocate % of gpu memory.
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def model_compiler(raw_model):
    adam = optimizers.adam(lr=config["RNN"].getfloat("initial_lr"), beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    logger.log(logging.DEBUG, f'Optimizer set to Adam with lr = {config["RNN"].getfloat("initial_lr")}')
    raw_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


def get_model(seq_length, features, dimout, config):
    regularizer = None
    logger.log(logging.DEBUG, f'Regularizer initialized to None')
    if config["RNN"].getfloat("l2_regularization") != 0:
        regularizer = regularizers.l2(l=config["RNN"].getfloat("l2_regularization"))
        logger.log(logging.DEBUG, f'Regularizer set to L2 = {config["RNN"].getfloat("l2_regularization")}')

    logger.log(logging.DEBUG, f'Dropout rate is {config["RNN"].getfloat("dropout")}')
    logger.log(logging.DEBUG, f'LSTM units: {config["RNN"].getint("LSTM_units")}')

    model = Sequential()
    model.add(Dense(32, kernel_regularizer=regularizer, input_shape=(seq_length, features)))
    model.add(Bidirectional(LSTM(config["RNN"].getint("LSTM_units"), return_sequences=True)))
    model.add(Dropout(config["RNN"].getfloat("dropout")))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizer))
    model.add(Dropout(config["RNN"].getfloat("dropout")))
    model.add(BatchNormalization())
    model.add(Dense(dimout, activation='softmax'))
    model.summary(print_fn=lambda x: logger.log(logging.INFO, x))
    model.summary()
    model_compiler(model)
    return model
