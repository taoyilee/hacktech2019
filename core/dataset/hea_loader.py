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

    def get_label(self, record, start_idx, ending_idx, default_label=0):
        df = self.label_dataframe
        roi = df.loc[(df['Record'] == int(record))]  # rows of interest
        self.logger.log(logging.DEBUG, f"Accessing {record} from sample {start_idx} to {ending_idx}")
        self.logger.log(logging.DEBUG, "Rows of interest are: (showing first 10 rows if more rows are selected)")
        self.logger.log(logging.DEBUG, roi.iloc[:, :4])
        if len(roi) == 0:
            return default_label
        rows_start = roi.loc[roi['Start_Index'] <= start_idx]

        self.logger.log(logging.DEBUG, rows_start.iloc[:, :4])
        rows_end = roi.loc[ending_idx <= roi['End_Index']]
        self.logger.log(logging.DEBUG, rows_end.iloc[:, :4])
        srow = rows_start.iloc[0]
        erow = rows_end.iloc[0]

        for idx, row in rows_start.iterrows():
            if srow['Start_Index'] < row['Start_Index']:
                srow = row

        for idx, row in rows_end.iterrows():
            if erow['End_Index'] < row['End_Index']:
                erow = row

        if pd.DataFrame.equals(srow, erow):
            if srow['Arrhythmia'] == True:
                return 1
            else:
                return 0
        else:
            start_row_idx = len(roi)
            for idx, row in roi.iterrows():
                if idx < start_row_idx:
                    if row['Start_Index'] == srow['Start_Index']:
                        start_row_idx = idx
                if idx >= start_row_idx:
                    if row['Arrhythmia'] == True:
                        return 1
                    if row['End_Index'] == erow['End_Index']:
                        break

        return 0

    def get_record_segment(self, record_name, start_idx, ending_idx, default_label=0):
        record = self.get_record(record_name)
        label = self.get_label(record_name, start_idx, ending_idx, default_label)
        return record.p_signal[start_idx:ending_idx, :], label

    def __repr__(self):
        return f"Excel HEA loader using {self.label}"
