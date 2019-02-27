import os
import logging
from typing import List, Dict
from keras.utils import Sequence
import numpy as np
from core.dataset.preprocessing import ECGRecordTicket, ECGDataset
from core.augmenters import AWGNAugmenter, RndInvertAugmenter, RndScaleAugmenter, RndDCAugmenter
from core.util.logger import LoggerFactory
import configparser as cp

config = cp.ConfigParser()
config.read("config.ini")


class BatchGenerator(Sequence):
    awgn_augmenter = None
    rndinv_augmenter = None
    rndscale_augmenter = None
    rnddc_augmenter = None

    def make_record_dict(self) -> Dict:
        rec_dict = {}
        for i in range(len(self.batch_numbers)):
            rec_dict.update({k + sum(self.batch_numbers[:i]): (k, self.dataset.tickets[i]) for k in
                             range(self.batch_numbers[i])})
        return rec_dict

    def __init__(self, dataset: ECGDataset, config, enable_augmentation=False, logger=None):
        """

        :param tickets: A list holding the recordnames of each record
        :param num_segments: A list holding total number segments from each record
        :param batch_size:
        """
        self.config = config
        self.logger = LoggerFactory(config).get_logger(logger_name=logger)

        self.dataset = dataset
        self.segment_length = config["preprocessing"].getint("sequence_length")
        self.logger.log(logging.INFO, f"Sequence length is {self.segment_length}")

        self.batch_size = config["preprocessing"].getint("batch_size")
        self.logger.log(logging.INFO, f"Batch size is {self.batch_size}")

        self.batch_numbers = np.ceil(self.dataset.record_len / self.segment_length / self.batch_size).astype(int)
        self.logger.log(logging.INFO, f"Number of batches from each record are {self.batch_numbers}")
        self.logger.log(logging.INFO, f"Total # batches {sum(self.batch_numbers)}")

        self.record_dict = self.make_record_dict()
        self.logger.log(logging.INFO, f"Record dictionary: ")
        for k, v in self.record_dict.items():
            self.logger.log(logging.INFO, f"{k}: {v}")

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_awgn"):
            self.awgn_augmenter = AWGNAugmenter(self.config["preprocessing"].getfloat("rms_noise_power_percent"))
            self.logger.log(logging.DEBUG, f"AWGN augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.awgn_augmenter}")

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rndinvert"):
            self.rndinv_augmenter = RndInvertAugmenter(self.config["preprocessing"].getfloat("rndinvert_prob"))
            self.logger.log(logging.DEBUG, f"Random inversion augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.rndinv_augmenter}")

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rndscale"):
            self.rndscale_augmenter = RndScaleAugmenter(self.config["preprocessing"].getfloat("scale"),
                                                        self.config["preprocessing"].getfloat("scale_prob"))
            self.logger.log(logging.DEBUG, f"Random scaling augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.rndscale_augmenter}")

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rnddc"):
            self.rnddc_augmenter = RndDCAugmenter(self.config["preprocessing"].getfloat("dc"),
                                                  self.config["preprocessing"].getfloat("dc_prob"))
            self.logger.log(logging.DEBUG, f"Random Dc augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.rnddc_augmenter}")

    def __len__(self):
        return sum(self.batch_numbers)

    def dump_labels(self):
        labels = []
        for _, record_batch in self.record_dict.items():
            local_batch_index, record_ticket = record_batch  # type: int, ECGRecordTicket
            max_starting_idx = record_ticket.siglen - self.segment_length
            self.logger.log(logging.DEBUG, f"max_starting_idx of {record_ticket} is {max_starting_idx}")
            for i in range(self.batch_size):
                self.logger.log(logging.DEBUG, f"local batch #{local_batch_index + i} of {record_ticket}")
                starting_idx = min(max_starting_idx, (local_batch_index + i) * self.segment_length)
                self.logger.log(logging.DEBUG, f"Starting index = {starting_idx}")
                label = record_ticket.get_label(starting_idx, starting_idx + self.segment_length)
                labels.append(label)
        return np.array(labels)

    def __getitem__(self, idx):
        local_batch_index, record_ticket = self.record_dict[idx]  # type: int, ECGRecordTicket
        self.logger.log(logging.DEBUG, f"batch[{idx}] {local_batch_index} - {record_ticket}")
        batch_length = self.segment_length * self.batch_size

        record_name = os.path.splitext(os.path.basename(record_ticket.hea_file))[0]
        real_batch_size = int(np.ceil(batch_length / self.segment_length))
        batch_x = []
        labels = []
        for b in range(real_batch_size):
            start_idx = local_batch_index * batch_length + b * self.segment_length
            ending_idx = start_idx + self.segment_length
            if (start_idx > record_ticket.siglen) or (ending_idx > record_ticket.siglen):
                start_idx -= abs(record_ticket.num_batches * batch_length - record_ticket.siglen)
                ending_idx -= abs(record_ticket.num_batches * batch_length - record_ticket.siglen)
            segment, label = record_ticket.hea_loader.get_record_segment(record_name, start_idx, ending_idx)
            batch_x.append(segment)
            labels.append(label)
        batch_x = np.array(batch_x)
        labels = np.array(labels)

        if self.awgn_augmenter is not None:
            batch_x = self.awgn_augmenter.augment(batch_x)

        if self.rndinv_augmenter is not None:
            batch_x = self.rndinv_augmenter.augment(batch_x)

        if self.rndscale_augmenter is not None:
            batch_x = self.rndscale_augmenter.augment(batch_x)

        if self.rnddc_augmenter is not None:
            batch_x = self.rnddc_augmenter.augment(batch_x)

        return batch_x, labels
