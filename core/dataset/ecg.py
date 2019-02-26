import os
import logging
from typing import List, Dict
from keras.utils import Sequence
import numpy as np
import random
import wfdb
from scipy.signal import resample
from core.dataset.preprocessing import ECGRecordTicket, ECGDataset
from core.augmenters import AWGNAugmenter, RndInvertAugmenter, RndScaleAugmenter, RndDCAugmenter
from core.util.logger import LoggerFactory


class ECGAnnotatedSequenceAugmented(Sequence):

    def __init__(self, dataset: List["ECGTaggedPair"], random_time_scale_percent=20, sequence_length=1300,
                 total_training_segments=512, awgn_rms_percent=2, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_time_scale = random_time_scale_percent
        self.sequence_length = sequence_length
        self.total_training_segments = total_training_segments
        self.awgn_ratio = awgn_rms_percent / 100

    def __len__(self):
        return self.total_training_segments

    def __getitem__(self, idx):
        original_seq_length = [int(self.sequence_length * random.uniform(1 - self.random_time_scale / 100,
                                                                         1 + self.random_time_scale / 100)) for _ in
                               range(self.batch_size)]
        raw_sequences = [random.choice(self.dataset).get_random_segment(seq_len) for seq_len in original_seq_length]
        batch_x = np.array([resample(r.x, self.sequence_length) for r in raw_sequences])
        batch_x_rms = np.sqrt(np.mean(batch_x ** 2, axis=1))
        noise = np.random.normal(0, self.awgn_ratio * batch_x_rms,
                                 (self.sequence_length, batch_x_rms.shape[0], batch_x_rms.shape[1])).swapaxes(0, 1)
        batch_x = batch_x + noise
        batch_y = [resample(r.y, self.sequence_length) for r in raw_sequences]
        return batch_x, np.array(batch_y)


class BatchGenerator(Sequence):
    awgn_augmenter = None
    rndinv_augmenter = None
    rndscale_augmenter = None
    rnddc_augmenter = None

    def compute_num_batches(self) -> List:
        return_list = []
        for ticket in self.dataset.tickets:
            with open(ticket.hea_file) as myfile:
                head = [next(myfile) for _ in range(1)]
            sig_len = int(str.split(head[0])[0])
            length = int(np.ceil(sig_len / self.segment_length / self.batch_size))
            return_list.append(length)
        return return_list

    def make_record_dict(self) -> Dict:
        rec_dict = {}
        for i in range(len(self.num_batch_each_record)):
            rec_dict.update({k + sum(self.num_batch_each_record[:i]): (k, self.dataset.tickets[i]) for k in
                             range(self.num_batch_each_record[i])})
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

        self.num_batch_each_record = self.compute_num_batches()
        self.logger.log(logging.INFO, f"Number of batches from each record are {self.num_batch_each_record}")
        self.logger.log(logging.INFO, f"Total # batches {sum(self.num_batch_each_record)}")

        self.record_dict = self.make_record_dict()

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_awgn"):
            self.awgn_augmenter = AWGNAugmenter(self.config["preprocessing"].getfloat("rms_noise_power_percent"))
            self.logger.log(logging.DEBUG, f"AWGN augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.awgn_augmenter}")

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rndinvert"):
            self.rndinv_augmenter = RndInvertAugmenter(self.config["preprocessing"].getfloat("rndinvert_prob"))
            self.logger.log(logging.DEBUG, f"Random inversion augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.rndinv_augmenter}")

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rndscale"):
            self.rndscale_augmenter = RndScaleAugmenter(self.config["preprocessing"].getfloat("scale"), self.config["preprocessing"].getfloat("scale_prob"))
            self.logger.log(logging.DEBUG, f"Random scaling augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.rndscale_augmenter}")

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rnddc"):
            self.rnddc_augmenter = RndDCAugmenter(self.config["preprocessing"].getfloat("dc"), self.config["preprocessing"].getfloat("dc_prob"))
            self.logger.log(logging.DEBUG, f"Random Dc augmenter enabled")
            self.logger.log(logging.DEBUG, f"{self.rnddc_augmenter}")

    def __len__(self):
        return sum(self.num_batch_each_record)

    def __getitem__(self, idx):
        local_batch_index, record_ticket = self.record_dict[idx]  # type: int, ECGRecordTicket
        batch_length = self.segment_length * self.batch_size
        signal = wfdb.rdrecord(os.path.splitext(record_ticket.hea_file)[0]).p_signal[
                 local_batch_index * batch_length:(local_batch_index + 1) * batch_length]
        real_batch_size = int(np.ceil(len(signal) / self.segment_length))
        batch_x = [signal[b * self.segment_length:(b + 1) * self.segment_length] for b in range(real_batch_size - 1)]
        batch_x.append(signal[(real_batch_size - 2) * self.segment_length:(real_batch_size - 1) * self.segment_length])
        batch_x = np.array(batch_x)


        if self.awgn_augmenter is not None:
            batch_x = self.awgn_augmenter.augment(batch_x)
            batch_x = np.array(batch_x)

        if self.rndinv_augmenter is not None:
            batch_x = self.rndinv_augmenter.augment(batch_x)
            batch_x = np.array(batch_x)

        if self.rndscale_augmenter is not None:

            batch_x = self.rndscale_augmenter.augment(batch_x)
            batch_x = np.array(batch_x)

        if self.rnddc_augmenter is not None:
            batch_x = self.rnddc_augmenter.augment(batch_x)
            batch_x = np.array(batch_x)



        return batch_x, np.array([record_ticket.label for _ in range(real_batch_size)])

