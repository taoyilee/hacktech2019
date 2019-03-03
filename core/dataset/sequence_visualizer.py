from core import Action
from core.dataset import BatchGenerator
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np


class SequenceVisualizer(Action):
    def __init__(self, config, experiment_env):
        super(SequenceVisualizer, self).__init__(config, experiment_env)

    def csv(self, dataset_generator: BatchGenerator):
        for i, mini_batch in enumerate(dataset_generator):
            csv_out = os.path.join(self.experiment_env.output_dir,
                                   "%s_batch_%03d.csv" % (dataset_generator.dataset.name, i))
            print("Writing csv to", csv_out)
            x = mini_batch[0]  # type:np.ndarray
            x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
            label = mini_batch[1][:, np.newaxis]  # type:np.ndarray
            x_label = np.concatenate((label, x), axis=1)
            np.savetxt(csv_out, x_label, delimiter=",")

    def visualize(self, dataset_generator: BatchGenerator, batch_limit=None, segment_limit=None):
        """

        :param dataset_generator:
        :param batch_limit: Visualize first "batch_limit" batches on;y
        :param segment_limit:   Visualize first "segment_limit" segments in each batch only
        :return:
        """
        matplotlib.use('PDF')
        pdf_out_0 = os.path.join(self.experiment_env.output_dir, f"{dataset_generator.dataset.name}_view_2.pdf")
        pdf_out_1 = os.path.join(self.experiment_env.output_dir, f"{dataset_generator.dataset.name}_view_1.pdf")
        print(f"Writing pdf visualization to {pdf_out_1}")
        with PdfPages(pdf_out_1) as pdf_1:
            with PdfPages(pdf_out_0) as pdf_0:
                print(f"Total # of batches {len(dataset_generator)}")
                for i, mini_batch in enumerate(dataset_generator):
                    if batch_limit is not None and i > batch_limit:
                        break
                    x = mini_batch[0]
                    label = mini_batch[1]
                    record_name = mini_batch[2]
                    start_index = mini_batch[3]
                    for j in range(x.shape[0]):
                        if segment_limit is not None and j > segment_limit:
                            break
                        plt.plot(x[j, :, 0], label=f"Lead 1")
                        plt.plot(x[j, :, 1], label=f"Lead 2")
                        plt.title(f"{dataset_generator.dataset.name} minibatch #{i}, segment #{j} - label: {label[j]}")
                        mitdb_tag = self.config["preprocessing"].get("MIT_DB_TAG")
                        nsrdb_tag = self.config["preprocessing"].get("NSR_DB_TAG")
                        plt.text(0, 0, "MIT_DB: {mitdb_tag}; NSR_DB: {nsrdb_tag}")
                        plt.text(0, 0.5, "Record_Name: {record_name}; Start_Index: {start_index[j]}")
                        plt.grid()
                        plt.legend()
                        if label[j] == 1:
                            pdf_1.savefig()
                        else:
                            pdf_0.savefig()
                        plt.close()
