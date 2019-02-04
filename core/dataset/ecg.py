import os
import pickle
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


class ECGAnnotatedSequence(Sequence):

    def __init__(self, dataset: List["ECGTaggedPair"], batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = [tagged_pair.x for tagged_pair in self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = [tagged_pair.y for tagged_pair in self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]]
        return np.array(batch_x), np.array(batch_y)


class ECGDataset:
    def __init__(self, name, dataset: List["ECGTaggedPair"]):
        self.name = name
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[item]

    @property
    def features(self):
        return self.dataset[0].x.shape[1]

    @property
    def total_samples(self):
        total = 0
        for d in self.dataset:
            total += len(d)
        return total

    def __len__(self):
        return len(self.dataset)

    @classmethod
    def from_pickle(cls, pickle_file) -> "ECGDataset":
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

    def save(self, output_dir):
        with open(os.path.join(output_dir, f"{self.name}.pickle"), 'wb') as f:
            pickle.dump(self, f)

    def to_exploded_set(self, sequence_length=1300, overlap_percent=5):
        logger.log(logging.INFO, f"dataset length: {len(self)}")
        logger.log(logging.INFO, f"total_samples: {self.total_samples}")
        logger.log(logging.INFO, f"dataset: {self.dataset}")
        return [s for d in self.dataset for s in d.explode(sequence_length, overlap_percent)]

    def to_sequence_generator(self, sequence_length=1300, overlap_percent=5, batch_size=32):
        exploded_set = self.to_exploded_set(sequence_length, overlap_percent)
        return ECGAnnotatedSequence(exploded_set, batch_size=batch_size)

    def to_sequence_generator_augmented(self, random_time_scale_percent=10, sequence_length=1300, dilation_factor=5,
                                        awgn_rms_percent=2, batch_size=32):
        total_training_segments = int(dilation_factor * self.total_samples / sequence_length / batch_size)
        logger.log(logging.INFO, f"total_training_segments = {total_training_segments}")
        return ECGAnnotatedSequenceAugmented(self.dataset, random_time_scale_percent, sequence_length,
                                             total_training_segments, awgn_rms_percent, batch_size)


class ECGTaggedPair:
    def __init__(self, x, y, fs, record_name):
        self.x = x
        self.y = y
        self.fs = fs
        self.record_name = record_name

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        logger.log(logging.DEBUG, f"slicing by {item}")
        return ECGTaggedPair(self.x[item], self.y[item], self.fs, self.record_name)

    def __repr__(self):
        return f"ECG Tagged Pair of size ({self.x.shape}, {self.y.shape}) fs@{self.fs} from {self.record_name}"

    def get_random_segment(self, sequence_length=1300):
        starting_index = random.randint(0, len(self) - sequence_length)
        return self[starting_index:starting_index + sequence_length]

    def explode(self, sequence_length=1300, overlap_percent=5):
        sequence_length = int(sequence_length)
        overlap_samples = sequence_length * overlap_percent // 100
        overlap_length = int(sequence_length - overlap_samples)
        segments = int(np.floor((len(self) - overlap_samples) / sequence_length) + 1)
        logger.log(logging.DEBUG, f"original length: {len(self)}")
        logger.log(logging.DEBUG, f"exploding by {sequence_length}/{overlap_length} segments: {segments}")
        exploded_segments = [self[overlap_length * i:overlap_length * i + sequence_length] for i in range(segments - 1)]
        exploded_segments.append(self[-sequence_length:])
        return exploded_segments
