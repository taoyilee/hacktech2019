import glob
import logging
import os
import pickle
import random
from os.path import join
from typing import Dict

import numpy as np
from tensorflow.keras.utils import Sequence

from core import Action
from core.augmenters import AWGNAugmenter, RndInvertAugmenter, RndScaleAugmenter, RndDCAugmenter
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
            self.experiment_env.write_json()
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
        self.logger.log(logging.INFO, "Sequence length is %d" % self.segment_length)

        self.batch_size = config["preprocessing"].getint("batch_size")
        self.logger.log(logging.INFO, "Batch size is %d" % self.batch_size)

        self.batch_length = self.batch_size * self.segment_length
        self.batch_numbers = np.ceil(self.dataset.record_len / self.segment_length / self.batch_size).astype(int)
        self.logger.log(logging.INFO, "Number of batches from each record are %s" % self.batch_numbers)
        self.logger.log(logging.INFO, "Total # batches %d", sum(self.batch_numbers))

        self.record_dict = self.make_record_dict()
        self.logger.log(logging.INFO, "Record dictionary: ")
        for k, v in self.record_dict.items():
            self.logger.log(logging.INFO, "%s : %s" % (k, v))

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_awgn"):
            self.awgn_augmenter = AWGNAugmenter(self.config["preprocessing"].getfloat("rms_noise_power_percent"))
            self.logger.log(logging.DEBUG, "AWGN augmenter enabled")
            self.logger.log(logging.DEBUG, self.awgn_augmenter)

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rndinvert"):
            self.rndinv_augmenter = RndInvertAugmenter(self.config["preprocessing"].getfloat("rndinvert_prob"))
            self.logger.log(logging.DEBUG, "Random inversion augmenter enabled")
            self.logger.log(logging.DEBUG, self.rndinv_augmenter)

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rndscale"):
            self.rndscale_augmenter = RndScaleAugmenter(self.config["preprocessing"].getfloat("scale"),
                                                        self.config["preprocessing"].getfloat("scale_prob"))
            self.logger.log(logging.DEBUG, "Random scaling augmenter enabled")
            self.logger.log(logging.DEBUG, self.rndscale_augmenter)

        if enable_augmentation and self.config["preprocessing"].getboolean("enable_rnddc"):
            self.rnddc_augmenter = RndDCAugmenter(self.config["preprocessing"].getfloat("dc"),
                                                  self.config["preprocessing"].getfloat("dc_prob"))
            self.logger.log(logging.DEBUG, "Random Dc augmenter enabled")
            self.logger.log(logging.DEBUG, self.rnddc_augmenter)

    def __len__(self):
        return sum(self.batch_numbers)

    def dump_labels(self):
        labels = []
        for _, record_batch in self.record_dict.items():
            local_batch_index, record_ticket = record_batch  # type: int, ECGRecordTicket
            max_starting_idx = record_ticket.siglen - self.segment_length
            starting_idx = min(max_starting_idx, local_batch_index * self.batch_length)
            ending_idx = min(record_ticket.siglen, starting_idx + self.segment_length * self.batch_size)
            max_starting_idx = record_ticket.siglen - self.segment_length
            real_batch_size = np.ceil((ending_idx - starting_idx) / self.segment_length).astype(int)
            for b in range(real_batch_size):
                b_start_idx = min(max_starting_idx, starting_idx + b * self.segment_length)
                b_ending_idx = min(record_ticket.siglen, b_start_idx + self.segment_length)
                segment, label = record_ticket.hea_loader.get_record_segment(record_ticket.record_name, b_start_idx,
                                                                             b_ending_idx)
                labels.append(label)
        return np.array(labels)

    def __getitem__(self, idx):
        local_batch_index, record_ticket = self.record_dict[idx]  # type: int, ECGRecordTicket
        max_starting_idx = record_ticket.siglen - self.segment_length
        starting_idx = min(max_starting_idx, local_batch_index * self.batch_length)
        ending_idx = min(record_ticket.siglen, starting_idx + self.segment_length * self.batch_size)
        real_batch_size = np.ceil((ending_idx - starting_idx) / self.segment_length).astype(int)
        batch_x = []
        labels = []
        for b in range(real_batch_size):
            b_start_idx = min(max_starting_idx, starting_idx + b * self.segment_length)
            b_ending_idx = min(record_ticket.siglen, b_start_idx + self.segment_length)
            segment, label = record_ticket.hea_loader.get_record_segment(record_ticket.record_name, b_start_idx,
                                                                         b_ending_idx)
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
        print(batch_x.shape, label.shape)
        #print(labels.shape)
        return batch_x, labels
