from core.dataset.ltstdb_hea_parser import LtstdbHeaParser
from core.dataset.ltstdb_comment_parser import LtstdbCommentParser
from core.dataset.ltstdb_comment_transformer import LtstdbCommentTransformer
from core.dataset.ltstdb_hea_transformer import LtstdbHeaTransformer
from core.dataset.ltstdb_hea_visitor import LtstdbHeaVisitor
from core.signal.signal_spec import SignalSpec
from typing import List
import configparser as cp
import os
from core.signal.ltstdb_signal import LtstdbSignal
import numpy as np


class LtstdbHea:
    name = None
    age = None  # type: int
    sex = True  # True: male, False: female
    associated_dat = []
    diagnoses = None
    treatment = None
    history = None
    comments = None
    _signals = None
    sampling_freq = 250
    number_of_samples_per_signal = None

    def __init__(self):
        self.signal_spec = []  # type:List[SignalSpec]
        config = cp.ConfigParser()
        config.read("config.ini")
        self.dat_dir = os.path.join(config["DEFAULT"].get("dataset_dir"), config["DEFAULT"].get("dat_dir"))

    @property
    def signals(self):
        self._read_signals()
        return self._signals

    @property
    def sampling_period(self):
        return 1 / self.sampling_freq

    @property
    def timestamps(self):
        return self.sampling_period * np.arange(0, self.number_of_samples_per_signal)

    @classmethod
    def from_file(cls, hea_file):
        parser = LtstdbHeaParser()
        comment_parser = LtstdbCommentParser()
        tree = parser.parse_file(hea_file)
        comments = parser.extract_comments(hea_file)
        comment_tree = comment_parser.parse(comments)
        new_instance = cls()
        LtstdbCommentTransformer(new_instance).transform(comment_tree)
        tree = LtstdbHeaTransformer(new_instance).transform(tree)
        LtstdbHeaVisitor(new_instance).visit(tree)
        return new_instance

    def _read_signals(self):
        if self._signals is None:
            dat_file_full_path = os.path.join(self.dat_dir, self.signal_spec[0].dat_name + ".dat")
            self._signals = LtstdbSignal.from_file(dat_file_full_path, num_signals=len(self.signal_spec),
                                                   format=self.signal_spec[0].signal_format)
