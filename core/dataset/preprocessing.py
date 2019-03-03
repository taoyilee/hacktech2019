import os
from core.dataset.hea_loader import HeaLoader
from os.path import join
import glob
import numpy as np
import pickle
import random
from typing import List


class ECGDataset:
    def set_hea(self):
        for ticket in self.tickets:
            ticket.use_hea = True
            ticket.hea_loader.config["preprocessing"]["use_hea"] = "True"
            ticket.hea_loader.use_hea = True

    def fix_path(self, mitdb_root="", nsrdb_root=""):
        for ticket in self.tickets:
            if "mitdb" in ticket.hea_file:
                ticket.hea_file = os.path.join(mitdb_root, os.path.splitext(os.path.split(ticket.hea_file)[1])[0])
            if "nsrdb" in ticket.hea_file:
                ticket.hea_file = os.path.join(nsrdb_root, os.path.splitext(os.path.split(ticket.hea_file)[1])[0])

    def __init__(self, name=""):
        self.tickets = []  # type:List[ECGRecordTicket]
        self.name = name

    @property
    def record_len(self) -> np.ndarray:
        return np.array([t.siglen for t in self.tickets])

    @classmethod
    def from_directory(cls, dataset_directory, hea_loader):
        new_instance = cls(name=os.path.split(dataset_directory)[1])
        files = glob.glob(join(dataset_directory, "*.npy"))
        new_instance.tickets = [ECGRecordTicket.from_hea_filenames(hea_filename, hea_loader) for hea_filename in
                                files]
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
            sliced_instance.name = "%s_%d_%d" % (self.name, index.start, index.stop)
            return sliced_instance
        else:  # single element
            sliced_instance.tickets = [self.tickets[index]]
            sliced_instance.name = "%s_%d" % (self.name, index)
            return sliced_instance

    def __repr__(self):
        return "%s has %d records" % (self.name, self.__len__())

    def __len__(self):
        return len(self.tickets)

    @classmethod
    def from_pickle(cls, pickle_file) -> "ECGDataset":
        with open(pickle_file, "rb") as f:
            return pickle.load(f)

    def save(self, output_dir):
        with open(os.path.join(output_dir, "%s.pickle" % self.name), 'wb') as f:
            pickle.dump(self, f)
        return os.path.join(output_dir, "%s.pickle" % self.name)


class ECGRecordTicket:
    _siglen = None
    hea_loader = None  # type:HeaLoader

    def __init__(self):
        self.use_hea = False
        self.hea_file = ""
        self.label = None
        self.num_batches = 0
        self.max_index = None

    def get_label(self, start_idx, end_idx):
        return self.hea_loader.get_label(self.record_name, start_idx, end_idx)

    def get_signal(self, start_idx, end_idx):
        return self.hea_loader.get_record(self.record_name).p_signal[start_idx:end_idx, :]

    @property
    def record_name(self):
        return os.path.splitext(os.path.basename(self.hea_file))[0]

    @property
    def siglen(self) -> int:
        if self.max_index is not None:
            return self.max_index

        if self._siglen is None:
            if self.use_hea:
                with open(self.hea_file) as hea_fptr:
                    head = [next(hea_fptr) for _ in range(1)]
                    self._siglen = int(str.split(head[0])[3])
            else:
                self._siglen = np.load(self.hea_file).shape[0]
        return self._siglen

    @classmethod
    def from_hea_filenames(cls, hea, hea_loader):
        new_instance = cls()
        new_instance.hea_file = hea
        new_instance.hea_loader = hea_loader
        return new_instance

    def __repr__(self):
        return "%s %s" % (self.hea_file, self.hea_loader)


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
        return "ECG Tagged Pair of size ({self.x.shape}, {self.y.shape}) from {self.record_name}"

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
