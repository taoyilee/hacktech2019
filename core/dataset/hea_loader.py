import pandas as pd
from abc import ABC, abstractmethod
import wfdb
import os
from functools import lru_cache


class HeaLoader(ABC):
    signal = []
    hea_directory = ""

    def __init__(self, hea_directory, label):
        """

        :param hea_directory: The directory which contains *.hea and *.dat
        :param label:  either 0, 1
        """
        self.label = label
        self.hea_directory = hea_directory

    @classmethod
    def load(cls, hea_directory, label):
        if isinstance(label, int):
            return HeaLoaderFixedLabel(hea_directory, label)
        else:
            return HeaLoaderExcel(hea_directory, label)

    @lru_cache(maxsize=48)
    def get_record(self, record_name):
        record_name = os.path.splitext(record_name)[0]
        return wfdb.rdrecord(os.path.join(self.hea_directory, record_name))

    @abstractmethod
    def get_record_segment(self, record_name, start_idx, ending_idx):
        pass


class HeaLoaderFixedLabel(HeaLoader):
    def __init__(self, hea_directory, label):
        """

                :param hea_directory: The directory which contains *.hea and *.dat
                :param label:  either 0, 1
                """
        super(HeaLoaderFixedLabel, self).__init__(hea_directory, label)

    def get_record_segment(self, record_name, start_idx, ending_idx):
        record = self.get_record(record_name)
        return record.p_signal[start_idx:ending_idx, :], self.label


class HeaLoaderExcel(HeaLoader):
    def __init__(self, hea_directory, excel_path):
        """

                :param hea_directory: The directory which contains *.hea and *.dat
                :param excel_path:  an Excel spread sheet
                """
        super(HeaLoaderExcel, self).__init__(hea_directory, excel_path)
        self.label_dataframe = pd.read_excel(excel_path)

    def get_label(self, record, start_idx, ending_idx):
        return self.label_dataframe

    def get_record_segment(self, record_name, start_idx, ending_idx):
        record = self.get_record(record_name)
        return record.p_signal[start_idx:ending_idx, :], self.label
