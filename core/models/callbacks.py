from keras.callbacks import Callback
import os
from sklearn.metrics import roc_auc_score
import numpy as np
from core.dataset.ecg import BatchGenerator


def computer_roc(model, y, data_generator):
    roc_auc = roc_auc_score(y, model.predict_generator(data_generator, verbose=0))
    return roc_auc


class SaveModel(Callback):

    def __init__(self, model, output_dir):
        super(SaveModel, self).__init__()
        self.model = model
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(os.path.join(self.output_dir, f"weights_{epoch + 1:02d}.h5"))


class ROCAUCCallback(Callback):
    def __init__(self, train_generator: BatchGenerator, validation_generator: BatchGenerator):
        print(f"Setting up ROC-AUC callback")
        super(ROCAUCCallback, self).__init__()
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.y_train = self.train_generator.dump_labels()
        self.y_val = self.validation_generator.dump_labels()

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc'] = computer_roc(self.model, self.y_train, self.train_generator)
        logs['roc_auc_val'] = computer_roc(self.model, self.y_val, self.validation_generator)
        print(f"ROC_AUC(Training): {logs['roc_auc']:.2f} - ROC_AUC(Dev): {logs['roc_auc_val']:.2f}")
