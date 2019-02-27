import pandas as pd
import logging
from abc import ABC, abstractmethod
import wfdb
import os
from functools import lru_cache
from core.util.logger import LoggerFactory


class HeaLoader(ABC):
    signal = []
    hea_directory = ""

    def __init__(self, hea_directory, label, logger=None):
        """

        :param hea_directory: The directory which contains *.hea and *.dat
        :param label:  either 0, 1
        """
        self.logger = logger if logger is not None else LoggerFactory.dummy()
        self.label = label
        self.hea_directory = hea_directory

    @classmethod
    def load(cls, hea_directory, label, logger=None):
        if isinstance(label, int):
            return HeaLoaderFixedLabel(hea_directory, label, logger)
        else:
            return HeaLoaderExcel(hea_directory, label, logger)

    @lru_cache(maxsize=48)
    def get_record(self, record_name):
        record_name = os.path.splitext(record_name)[0]
        return wfdb.rdrecord(os.path.join(self.hea_directory, record_name))

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
        # Restore the previously opened file's state. To do so, we need to
        # # reopen it and read from it until the line count is restored.
        # file = open(self.filename)
        # for _ in range(self.lineno):
        #     file.readline()
        # # Finally, save the file.
        # self.file = file


class HeaLoaderFixedLabel(HeaLoader):
    def __init__(self, hea_directory, label, logger=None):
        """

                :param hea_directory: The directory which contains *.hea and *.dat
                :param label:  either 0, 1
                """
        super(HeaLoaderFixedLabel, self).__init__(hea_directory, label, logger)

    def get_record_segment(self, record_name, start_idx, ending_idx):
        record = self.get_record(record_name)
        return record.p_signal[start_idx:ending_idx, :], self.label

    def __repr__(self):
        return f"Fixed label loader using label: {self.label}"

    def get_label(self, record_name, start_idx, ending_idx):
        return self.label


class HeaLoaderExcel(HeaLoader):
    def __init__(self, hea_directory, excel_path, logger=None):
        """

                :param hea_directory: The directory which contains *.hea and *.dat
                :param excel_path:  an Excel spread sheet
                """
        super(HeaLoaderExcel, self).__init__(hea_directory, excel_path, logger)
        if not os.path.isfile(excel_path):
            raise FileNotFoundError(f"Excel spreadsheet {excel_path} is not found.")
        self.label_dataframe = pd.read_excel(excel_path)
        self.data_dataframe_normal = self.label_dataframe.loc[~self.label_dataframe['Arrhythmia']]
        self.logger.log(logging.DEBUG, f"Data frame normal:")
        self.logger.log(logging.DEBUG, f"{self.data_dataframe_normal}")

    @lru_cache(maxsize=None)
    def get_roi(self, record):
        return self.data_dataframe_normal.loc[self.data_dataframe_normal['Record'] == int(record)]

    def get_label(self, record, start_idx, ending_idx):
        self.logger.log(logging.DEBUG, f"Accessing {record} from sample {start_idx} to {ending_idx}")
        roi = self.get_roi(record)
        if len(roi) == 0:  # no normal frames selected
            return 1
        row_start = roi.loc[(roi['Start_Index'] <= start_idx) & (roi['End_Index'] >= start_idx)]
        row_end = roi.loc[
            (roi['Start_Index'] <= ending_idx) & (roi['End_Index'] >= ending_idx)]  # type: pd.DataFrame
        row_between = roi.loc[
            (roi['Start_Index'] >= start_idx) & (roi['End_Index'] <= ending_idx)]  # type: pd.DataFrame
        row_relevant = row_start.append(row_between).append(row_end).drop_duplicates()
        if len(row_relevant) == 0:
            return 1
        elif len(row_relevant) == 1:
            return 0
        else:
            for rowi, rowj in zip(row_relevant.iloc[:-1].iterrows(),
                                  row_relevant.iloc[1:].iterrows()):
                if rowj[1].loc['Start_Index'] != rowi[1].loc['End_Index'] + 1:
                    return 1
            return 0

    def get_record_segment(self, record_name, start_idx, ending_idx):
        record = self.get_record(record_name)
        label = self.get_label(record_name, start_idx, ending_idx)
        return record.p_signal[start_idx:ending_idx, :], label

    def __repr__(self):
        return f"Excel HEA loader using {self.label}"
