import glob
import os
import pickle
import random
from os.path import join

import numpy as np

from core import Action
from core.dataset.ecg import BatchGenerator
from core.dataset.hea_loader import HeaLoader
from core.util.logger import LoggerFactory


class Preprocessor(Action):
    mitdb = None
    nsrdb = None

    def __init__(self, config, experiment_env):
        super(Preprocessor, self).__init__(config, experiment_env)
        if not self.experiment_env.resume:
            self.init_ecg_datasets()

    def init_ecg_datasets(self, logger=None):
        mitdb_path, nsrdb_path = self.config["mitdb"].get("dataset_npy_path"), self.config["nsrdb"].get(
            "dataset_npy_path")
        if not os.path.isdir(mitdb_path):
            raise FileNotFoundError("dataset directory %s does not exist" % mitdb_path)
        if not os.path.isdir(nsrdb_path):
            raise FileNotFoundError("dataset directory %s does not exist" % nsrdb_path)
        if logger is None:
            heaLoader_mit = HeaLoader.load(self.config, mitdb_path, self.config["mitdb"].get("excel_label"))
            heaLoader_nsr = HeaLoader.load(self.config, nsrdb_path,
                                           self.config["preprocessing"].getint("NSR_DB_TAG"))
        else:
            heaLoader_mit = HeaLoader.load(self.config, mitdb_path, self.config["mitdb"].get("excel_label"),
                                           logger=LoggerFactory(self.config).get_logger(logger_name="mit_loader"))
            heaLoader_nsr = HeaLoader.load(self.config, nsrdb_path,
                                           self.config["preprocessing"].getint("NSR_DB_TAG"),
                                           logger=LoggerFactory(self.config).get_logger(logger_name="nsr_loader"))
        self.mitdb = ECGDataset.from_directory(mitdb_path, heaLoader_mit)
        self.nsrdb = ECGDataset.from_directory(nsrdb_path, heaLoader_nsr)

    def split_dataset(self, mitdb: "ECGDataset", nsrdb: "ECGDataset"):
        mitdb.shuffle()
        nsrdb.shuffle()
        mixture_db = mitdb + nsrdb
        mixture_db.name = "mixture_db"

        dev_record_each = self.config["preprocessing"].getint("dev_record_each")
        test_record_each = self.config["preprocessing"].getint("test_record_each")
        dev_slice = slice(None, dev_record_each)
        test_slice = slice(dev_record_each, dev_record_each + test_record_each)
        train_slice = slice(dev_record_each + test_record_each, None)
        nsrdb_slice = nsrdb[dev_slice]
        for ticket in nsrdb_slice.tickets:
            ticket.max_index = int(
                ticket.siglen * .26)  # .26 is how much of the signal we want to keep so we don't have an imbalanced dev set

        dev_set = mitdb[dev_slice] + nsrdb[dev_slice]  # type: ECGDataset
        dev_set.name = "development_set"
        test_set = mitdb[test_slice] + nsrdb[test_slice]  # type: ECGDataset
        test_set.name = "test_set"
        train_set = mitdb[train_slice] + nsrdb[train_slice]  # type: ECGDataset
        train_set.name = "training_set"
        train_set.print()
        dev_set.print()
        test_set.print()

        return train_set, dev_set, test_set

    def preprocess(self):
        if self.experiment_env.resume:
            print("Resuming datasets from", self.experiment_env.training_set)
            train_set = ECGDataset.from_pickle(self.experiment_env.training_set)
            print("Resuming datasets from", self.experiment_env.development_set)
            dev_set = ECGDataset.from_pickle(self.experiment_env.development_set)
        else:
            train_set, dev_set, test_set = self.split_dataset(self.mitdb, self.nsrdb)
            self.experiment_env.add_key(
                **{d.name: d.save(self.experiment_env.output_dir) for d in [train_set, dev_set, test_set]})
        train_generator = BatchGenerator(train_set, self.config, enable_augmentation=True, logger="train_sequencer")
        dev_generator = BatchGenerator(dev_set, self.config, enable_augmentation=False, logger="dev_sequencer")
        return train_generator, dev_generator


class ECGDataset:
    def print(self):
        print("tickets in %s" % self.name)
        for ticket in self.tickets:
            print(ticket)

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
            sliced_instance.name = "%s_%s_%s" % (self.name, index.start, index.stop)
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
