import os
import random
from os.path import join
import glob
import numpy as np


class ECGDataset:
    # currently, only takes directory path as input

    def __init__(self, name=""):
        self.tickets = []
        self.name = name

    @classmethod
    def from_directory(cls, dataset_directory, label):
        new_instance = cls(name=os.path.split(dataset_directory)[1])
        # files = [splitext(f)[0] for f in listdir(dataset_directory) if isfile(join(dataset_directory, f))]
        # files = list(set(files))
        # files.sort()
        files = glob.glob(join(dataset_directory, "*.hea"))
        new_instance.tickets = [ECGRecordTicket.from_hea_filenames(hea_filename, label) for hea_filename in files]
        # for f in files:
        #     try:
        #         # label for each record
        #         ecgtaggedpair = ECGTaggedPair(wfdb.rdrecord(join(dataset_directory, f)).p_signal, np.array([label]),
        #                                       join(dataset_directory, f))
        #         new_instance.dataset.append(ecgtaggedpair)
        #     except FileNotFoundError:
        #         logger.log(logging.INFO, f"{join(dataset_directory, f)} +  is not a record")
        return new_instance

    def __add__(self, object: "ECGDataset"):
        newECG = ECGDataset()
        newECG.tickets = self.tickets + object.tickets
        return newECG

    def __getitem__(self, index):
        sliced_instance = ECGDataset()
        if isinstance(index, slice):
            sliced_instance.tickets = self.tickets[index]
            return sliced_instance
        sliced_instance.tickets = [self.tickets[index]]
        return sliced_instance[index]

    def __repr__(self):
        return f"{self.name} has {self.__len__()} records"

    def __len__(self):
        return len(self.tickets)

    # def shuffle(self):
    #     x = [[d] for d in self.tickets]
    #     shuffle(x)
    #     self.tickets = [d[0] for d in x]
    #
    # @property
    # def features(self):
    #     return self.tickets[0].x.shape[1]
    #
    # @property
    # def total_samples(self):
    #     total = 0
    #     for d in self.tickets:
    #         total += len(d)
    #     return total
    #
    # @classmethod
    # def from_pickle(cls, pickle_file) -> "ECGDataset":
    #     with open(pickle_file, "rb") as f:
    #         return pickle.load(f)
    #
    # def save(self, output_dir):
    #     with open(os.path.join(output_dir, f"{self.name}.pickle"), 'wb') as f:
    #         pickle.dump(self, f)
    #
    # def to_exploded_set(self, sequence_length=1300, overlap_percent=5):
    #     logger.log(logging.INFO, f"dataset length: {len(self)}")
    #     logger.log(logging.INFO, f"total_samples: {self.total_samples}")
    #     logger.log(logging.INFO, f"dataset: {self.tickets}")
    #     return [s for d in self.tickets for s in d.explode(sequence_length, overlap_percent)]
    #
    # def to_sequence_generator(self, sequence_length=1300, overlap_percent=5, batch_size=32):
    #     exploded_set = self.to_exploded_set(sequence_length, overlap_percent)
    #     return ECGAnnotatedSequence(exploded_set, batch_size=batch_size)
    #
    # def to_sequence_generator_augmented(self, random_time_scale_percent=10, sequence_length=1300, dilation_factor=5,
    #                                     awgn_rms_percent=2, batch_size=32):
    #     total_training_segments = int(dilation_factor * self.total_samples / sequence_length / batch_size)
    #     logger.log(logging.INFO, f"total_training_segments = {total_training_segments}")
    #     return ECGAnnotatedSequenceAugmented(self.tickets, random_time_scale_percent, sequence_length,
    #                                          total_training_segments, awgn_rms_percent, batch_size)
    #


class ECGRecordTicket:
    @classmethod
    def from_hea_filenames(cls, hea, label):
        new_instance = cls()
        new_instance.hea_file = hea
        new_instance.label = label


class ECGTaggedPair:
    def __init__(self, x, y, record_name):
        self.x = x
        self.y = y  # label
        self.record_name = record_name

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        # logger.log(logging.DEBUG, f"slicing by {item}")
        return ECGTaggedPair(self.x[item], self.y[item], self.record_name)

    def __repr__(self):
        return f"ECG Tagged Pair of size ({self.x.shape}, {self.y.shape}) from {self.record_name}"

    def get_segment(self, start, end):
        # TODO: some range check
        return self.x[start:end]

    def get_random_segment(self, sequence_length=1300):
        starting_index = random.randint(0, len(self) - sequence_length)
        return self[starting_index:starting_index + sequence_length]

    def explode(self, sequence_length=1300, overlap_percent=5):
        sequence_length = int(sequence_length)
        overlap_samples = sequence_length * overlap_percent // 100
        overlap_length = int(sequence_length - overlap_samples)
        segments = int(np.floor((len(self) - overlap_samples) / sequence_length) + 1)
        # logger.log(logging.DEBUG, f"original length: {len(self)}")
        # logger.log(logging.DEBUG, f"exploding by {sequence_length}/{overlap_length} segments: {segments}")
        exploded_segments = [self[overlap_length * i:overlap_length * i + sequence_length] for i in range(segments - 1)]
        exploded_segments.append(self[-sequence_length:])
        return exploded_segments
