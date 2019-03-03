import os

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

from core.dataset.ecg import BatchGenerator


def computer_roc(model, y, data_generator):
    roc_auc = roc_auc_score(y, model.predict_generator(data_generator, verbose=0))
    return roc_auc


class EpochEnd(Callback):

    def __init__(self, experiment_env):
        super(EpochEnd, self).__init__()
        self.experiment_env = experiment_env

    def on_epoch_end(self, epoch, logs={}):
        logs["lr"] = self.model.optimizer.lr
        logs["epoch"] = epoch
        self.experiment_env.add_key(**{"lr": self.model.optimizer.lr, "epoch": epoch})
        self.experiment_env.write_json()


class SaveModel(Callback):

    def __init__(self, model, output_dir):
        super(SaveModel, self).__init__()
        self.model = model
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(os.path.join(self.output_dir, "weights_", str(epoch), ".h5"))


class ROCAUCCallback(Callback):
    def __init__(self, train_generator: BatchGenerator, validation_generator: BatchGenerator):
        print("Setting up ROC-AUC callback")
        super(ROCAUCCallback, self).__init__()
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.y_train = self.train_generator.dump_labels()
        self.y_val = self.validation_generator.dump_labels()

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc'] = computer_roc(self.model, self.y_train, self.train_generator)
        logs['roc_auc_val'] = computer_roc(self.model, self.y_val, self.validation_generator)
        print("ROC_AUC(Training): ", (logs['roc_auc']), "ROC_AUC(Dev): ", logs['roc_auc_val'])
