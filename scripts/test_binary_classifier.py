import configparser as cp
from core.util.experiments import ExperimentEnv
import os
import numpy as np
import argparse
from core import Tester, SequenceVisualizer
from core.dataset.preprocessing import ECGDataset
from core.dataset import BatchGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, help="configuration file")
    args = parser.parse_args()
    configuration_file = args.c

    if os.path.isfile(configuration_file):
        print("Using configuration file",configuration_file)
    else:
        raise FileNotFoundError("configuration file",configuration_file, "dose not exist")
    config = cp.ConfigParser()
    config.read(configuration_file)
    experiment_env = ExperimentEnv.setup_testing(config)
    test_set = ECGDataset.from_pickle(experiment_env.test_set)
    RNN_Tester = Tester(config, experiment_env, logger="tester")

    test_generator = BatchGenerator(test_set, config, enable_augmentation=False, logger="test_sequencer")
    if config["RNN-test"].getboolean("plot_datasets"):
        sv = SequenceVisualizer(config, experiment_env)
        sv.visualize(test_generator, batch_limit=None, segment_limit=15)
    if not os.path.isfile(os.path.join(experiment_env.output_dir, "yhat.npy")):
        yhat = RNN_Tester.predict(test_generator)
        np.save(os.path.join(experiment_env.output_dir, "yhat.npy"), yhat)
    else:
        yhat = np.load(os.path.join(experiment_env.output_dir, "yhat.npy"))
    y = np.array([j for i in test_generator for j in i[1].tolist()])[:, np.newaxis]

    plt.figure(figsize=(10, 6))
    fpr, tpr, thresholds = roc_curve(y, yhat)
    print("ROC AUC:", roc_auc_score(y, yhat))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(thresholds, tpr, color='blue')
    plt.ylabel('True Positive Rate')
    plt.xlabel('Threshold')
    plt.grid()
    plt.xlim([min(thresholds), 0.85])
    plt.ylim([0.0, 1])
    plt.title('Thresholds')
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_env.output_dir, "auc_roc.png"))
