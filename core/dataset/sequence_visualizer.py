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
                                   f"{dataset_generator.dataset.name}_batch_{i:03d}.csv")
            print(f"Writing csv to {csv_out}")
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
        pdf_out = os.path.join(self.experiment_env.output_dir, f"{dataset_generator.dataset.name}_view.pdf")
        print(f"Writing pdf visualization to {pdf_out}")
        with PdfPages(pdf_out) as pdf:
            print(f"Total # of batches {len(dataset_generator)}")
            for i, mini_batch in enumerate(dataset_generator):
                if batch_limit is not None and i > batch_limit:
                    break
                x = mini_batch[0]
                label = mini_batch[1]
                for j in range(x.shape[0]):
                    if segment_limit is not None and j > segment_limit:
                        break
                    plt.plot(x[j, :, 0], label=f"Lead 1")
                    plt.plot(x[j, :, 1], label=f"Lead 2")
                    plt.title(f"{dataset_generator.dataset.name} minibatch #{i}, segment #{j} - label: {label[j]}")
                    mitdb_tag = self.config["preprocessing"].get("MIT_DB_TAG")
                    nsrdb_tag = self.config["preprocessing"].get("NSR_DB_TAG")
                    plt.text(0, 0, f"MIT_DB: {mitdb_tag}; NSR_DB: {nsrdb_tag}")
                    plt.grid()
                    plt.legend()
                    pdf.savefig()
                    plt.close()
