import os
from os.path import join
import glob
import numpy as np
import pickle
import random


class ECGDataset:

    def __init__(self, name=""):
        self.tickets = []
        self.name = name

    @classmethod
    def from_directory(cls, dataset_directory, label):
        new_instance = cls(name=os.path.split(dataset_directory)[1])
        files = glob.glob(join(dataset_directory, "*.hea"))
        new_instance.tickets = [ECGRecordTicket.from_hea_filenames(hea_filename, label) for hea_filename in files]

        #FIXME: for fast debugging
        #new_instance = new_instance[1:2]

        return new_instance

    def shuffle(self):
        random.shuffle(self.tickets)

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

    @classmethod
    def from_pickle(cls, pickle_file) -> "ECGDataset":
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

    def save(self, output_dir):
        with open(os.path.join(output_dir, f"{self.name}.pickle"), 'wb') as f:
            pickle.dump(self, f)


class ECGRecordTicket:

    def __init__(self):
        self.hea_file = ""
        self.label = None

    @classmethod
    def from_hea_filenames(cls, hea, label):
        new_instance = cls()
        new_instance.hea_file = hea
        new_instance.label = label

        return new_instance

    def __repr__(self):
        return f"{self.hea_file} {self.label}"


class ECGTaggedPair:
    def __init__(self, x, y, record_name):
        self.x = x
        self.y = y  # label
        self.record_name = record_name

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
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
        exploded_segments = [self[overlap_length * i:overlap_length * i + sequence_length] for i in range(segments - 1)]
        exploded_segments.append(self[-sequence_length:])
        return exploded_segments
