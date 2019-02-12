import os
import matplotlib
import numpy as np
import configparser as cp
import logging
from keras.models import model_from_json
from keras.models import Model
from core.util.plotting import plotecg, plot_annotation
from matplotlib.backends.backend_pdf import PdfPages
from core.dataset.preprocessing import ECGDataset
import matplotlib.pyplot as plt
from core.models.rnn import model_compiler

matplotlib.use('PDF')
logger = logging.getLogger('test')
logger.setLevel(logging.DEBUG)
config = cp.ConfigParser()
config.read("config.ini")
try:
    os.makedirs(config["logging"].get("logdir"))
except FileExistsError:
    pass
fh = logging.FileHandler(os.path.join(config["logging"].get("logdir"), "test.log"), mode="w+")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":

    np.random.seed(0)
    config = cp.ConfigParser()
    config.read("config.ini")
    test_set_pickle = config["RNN-test"].get("test_set")
    weights = config["RNN-test"].get("weights")

    model_json = config["RNN-test"].get("model_json")
    logger.log(logging.INFO, f"Test set pickle: {test_set_pickle}")
    logger.log(logging.INFO, f"Trained weights: {weights}")
    test_set = ECGDataset.from_pickle(test_set_pickle)
    experiment_dir, _ = os.path.split(weights)
    logger.log(logging.INFO, f"Output dir: {experiment_dir}")
    logger.log(logging.INFO, f"Test set: {test_set} ({len(test_set)} records, {test_set.total_samples} samples)")
    test_generator = test_set.to_sequence_generator(config["RNN-train"].getint("sequence_length"), 0,
                                                    config["RNN-test"].getint("batch_size"))
    with open(model_json, "r") as f:
        model = model_from_json(f.read())  # type: Model
    model.load_weights(weights)
    model.summary(print_fn=lambda x: logger.log(logging.INFO, x))
    model.summary()
    model_compiler(model)

    loss, acc = model.evaluate_generator(test_generator, verbose=1)
    logger.log(logging.INFO, f'Test loss (categorical cross entropy): {loss} , Test accuracy: {acc}')
    print(f'Test loss (categorical cross entropy): {loss} , Test accuracy: {acc}')
    predicted_annotations = model.predict_generator(test_generator)
    raw_prediction_out = os.path.join(experiment_dir, 'predicted_annotations_raw.npy')
    logger.log(logging.INFO, f"predicted_annotations: {predicted_annotations.shape}")
    logger.log(logging.INFO, f"raw prediction output saved to {raw_prediction_out}")
    predicted_annotations.tofile(raw_prediction_out)
    argmax_predictions = np.argmax(predicted_annotations, axis=2)
    argmax_prediction_out = os.path.join(experiment_dir, 'predicted_annotations_argmax.npy')
    logger.log(logging.INFO, f"argmax predicted_annotations: {argmax_predictions.shape}")
    logger.log(logging.INFO, f"argmax prediction output saved to {argmax_prediction_out}")
    argmax_predictions.tofile(argmax_prediction_out)

    with PdfPages(os.path.join(experiment_dir, 'testing_set.pdf')) as pdf:
        for i, segment in enumerate(test_set.to_exploded_set()):
            plotecg(segment.x, segment.y, fs=segment.fs)
            plot_annotation(predicted_annotations[i], fs=segment.fs)
            plot_annotation(argmax_predictions[i], fs=segment.fs)
            logger.log(logging.INFO, f"argmax_label[{i}]: {np.argmax(segment.y, axis=1)}")
            logger.log(logging.INFO, f"argmax_predictions[{i}]: {argmax_predictions[i]}")
            plt.title(f"ECG Segment of {segment.record_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Normalized ECG Signal ")
            pdf.savefig()
            plt.close()
