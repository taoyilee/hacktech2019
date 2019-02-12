import os
from typing import List
from keras.utils import Sequence
import numpy as np
import logging
import configparser as cp
import random
from scipy.signal import resample

logger = logging.getLogger('ecg')
logger.setLevel(logging.INFO)
config = cp.ConfigParser()
config.read("config.ini")
try:
    os.makedirs(config["logging"].get("logdir"))
except FileExistsError:
    pass
fh = logging.FileHandler(os.path.join(config["logging"].get("logdir"), "ecg.log"), mode="w+")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

from core.dataset.preprocessing import ECGRecordTicket, ECGDataset


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

    def compute_num_segments(self) -> List:
        return []

    def __init__(self, dataset: ECGDataset, segment_length=1300, batch_size=32):
        """

        :param tickets: A list holding the recordnames of each record
        :param num_segments: A list holding total number segments from each record
        :param batch_size:
        """
        self.dataset = dataset
        self.num_segments = self.compute_num_segments()
        self.batch_size = batch_size

    def __len__(self):
        return int(sum(self.num_segments) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = [tagged_pair.x for tagged_pair in self.tickets[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = [tagged_pair.y for tagged_pair in self.tickets[idx * self.batch_size:(idx + 1) * self.batch_size]]
        return np.array(batch_x), np.array(batch_y)
