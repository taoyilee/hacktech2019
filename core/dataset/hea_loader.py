import pandas as pd
import numpy as np
import logging
from abc import ABC, abstractmethod
import wfdb
import os
from functools import lru_cache
from core.util.logger import LoggerFactory


class HeaLoader(ABC):
    signal = []
    hea_directory = ""

    def __init__(self, config, hea_directory, label, logger=None):
        """

        :param hea_directory: The directory which contains *.hea and *.dat
        :param label:  either 0, 1
        """
        self.config = config
        self.logger = logger if logger is not None else LoggerFactory.dummy()
        self.label = label
        self.hea_directory = hea_directory

    @classmethod
    def load(cls, config, hea_directory, label, logger=None):
        if isinstance(label, int):
            return HeaLoaderFixedLabel(config, hea_directory, label, logger)
        else:
            return HeaLoaderExcel(config, hea_directory, label, logger)

    @lru_cache(maxsize=48)
    def get_record(self, record_name):
        record_name = os.path.splitext(record_name)[0]
        print(os.path.join(self.hea_directory, record_name + ".npy"))
        return np.load(os.path.join(self.hea_directory, record_name + ".npy"))
        # return wfdb.rdrecord(os.path.join(self.hea_directory, record_name))

    @abstractmethod
    def get_record_segment(self, record_name, start_idx, ending_idx):
        pass

    @abstractmethod
    def get_label(self, record_name, start_idx, ending_idx):
        pass

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['logger']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        self.logger = LoggerFactory.dummy()


class HeaLoaderFixedLabel(HeaLoader):
    def __init__(self, config, hea_directory, label, logger=None):
        """

                :param hea_directory: The directory which contains *.hea and *.dat
                :param label:  either 0, 1
                """
        super(HeaLoaderFixedLabel, self).__init__(config, hea_directory, label, logger)

    def get_record_segment(self, record_name, start_idx, ending_idx):
        record = self.get_record(record_name)
        return record[start_idx:ending_idx, :], self.label

    def __repr__(self):
        return "Fixed label loader using label: %s" % self.label

    def get_label(self, record_name, start_idx, ending_idx):
        return self.label


class HeaLoaderExcel(HeaLoader):
    def __init__(self, config, hea_directory, excel_path, logger=None):
        """

                :param hea_directory: The directory which contains *.hea and *.dat
                :param excel_path:  an Excel spread sheet
                """
        super(HeaLoaderExcel, self).__init__(config, hea_directory, excel_path, logger)
        if not os.path.isfile(excel_path):
            raise FileNotFoundError("Excel spreadsheet {excel_path} is not found.")
        self.label_dataframe = pd.read_excel(excel_path)

    @lru_cache(maxsize=None)
    def get_roi(self, record):
        return self.label_dataframe.loc[self.label_dataframe['Record'] == int(record)]

    def get_label(self, record, start_idx, ending_idx):
        self.logger.log(logging.DEBUG, "Accessing {record} from sample {start_idx} to {ending_idx}")
        roi = self.get_roi(record)
        if len(roi) == 0:  # no normal frames selected
            raise ValueError("Record {record} is not found")
        row_start = roi.loc[(roi['Start_Index'] <= start_idx) & (roi['End_Index'] >= start_idx)]
        row_end = roi.loc[
            (roi['Start_Index'] <= ending_idx) & (roi['End_Index'] >= ending_idx)]  # type: pd.DataFrame
        row_between = roi.loc[
            (roi['Start_Index'] >= start_idx) & (roi['End_Index'] <= ending_idx)]  # type: pd.DataFrame
        row_relevant = row_start.append(row_between).append(row_end).drop_duplicates()

        arrhythmia_tag = self.config["preprocessing"].getint("MIT_DB_TAG")
        normal_tag = self.config["preprocessing"].getint("NSR_DB_TAG")
        return arrhythmia_tag if row_relevant["Arrhythmia"].any() else normal_tag

    def get_record_segment(self, record_name, start_idx, ending_idx):
        record = self.get_record(record_name)
        label = self.get_label(record_name, start_idx, ending_idx)
        return record[start_idx:ending_idx, :], label

    def __repr__(self):
        return "Excel HEA loader using %s" % self.label
