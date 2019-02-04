from keras.callbacks import Callback
import os


class SaveModel(Callback):

    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs={}):
        self.model.save(os.path.join(self.output_dir, f"weights_{epoch + 1:02d}.h5"))
