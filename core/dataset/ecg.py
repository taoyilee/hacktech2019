import os
from typing import List, Dict
from keras.utils import Sequence
import numpy as np
import random
import wfdb
from scipy.signal import resample
from core.dataset.preprocessing import ECGRecordTicket, ECGDataset


class ECGAnnotatedSequenceAugmented(Sequence):

    def __init__(self, dataset: List["ECGTaggedPair"], random_time_scale_percent=20, sequence_length=1300,
                 total_training_segments=512, awgn_rms_percent=2, batch_size=32):
        self.dataset = dataset
        self.batch_size = batch_size
        self.random_time_scale = random_time_scale_percent
        self.sequence_length = sequence_length
        self.total_training_segments = total_training_segments
        self.awgn_ratio = awgn_rms_percent / 100

    def __len__(self):
        return self.total_training_segments

    def __getitem__(self, idx):
        original_seq_length = [int(self.sequence_length * random.uniform(1 - self.random_time_scale / 100,
                                                                         1 + self.random_time_scale / 100)) for _ in
                               range(self.batch_size)]
        raw_sequences = [random.choice(self.dataset).get_random_segment(seq_len) for seq_len in original_seq_length]
        batch_x = np.array([resample(r.x, self.sequence_length) for r in raw_sequences])
        batch_x_rms = np.sqrt(np.mean(batch_x ** 2, axis=1))
        noise = np.random.normal(0, self.awgn_ratio * batch_x_rms,
                                 (self.sequence_length, batch_x_rms.shape[0], batch_x_rms.shape[1])).swapaxes(0, 1)
        batch_x = batch_x + noise
        batch_y = [resample(r.y, self.sequence_length) for r in raw_sequences]
        return batch_x, np.array(batch_y)


class BatchGenerator(Sequence):

    def compute_num_batches(self) -> List:
        return_list = []
        for ticket in self.dataset.tickets:
            with open(ticket.hea_file) as myfile:
                head = [next(myfile) for _ in range(1)]
            sig_len = int(str.split(head[0])[0])
            length = int(np.ceil(sig_len / self.segment_length / self.batch_size))
            return_list.append(length)
        return return_list

    def make_record_dict(self) -> Dict:
        rec_dict = {}
        for i in range(len(self.num_batch_each_record)):
            rec_dict.update({k + sum(self.num_batch_each_record[:i]): (k, self.dataset.tickets[i]) for k in
                             range(self.num_batch_each_record[i])})
        return rec_dict

    def __init__(self, dataset: ECGDataset, segment_length=1300, batch_size=32):
        """

        :param tickets: A list holding the recordnames of each record
        :param num_segments: A list holding total number segments from each record
        :param batch_size:
        """
        self.dataset = dataset
        self.segment_length = segment_length
        self.batch_size = batch_size
        self.num_batch_each_record = self.compute_num_batches()
        self.record_dict = self.make_record_dict()

    def __len__(self):
        return sum(self.num_batch_each_record)

    def __getitem__(self, idx):
        local_batch_index, record_ticket = self.record_dict[idx]  # type: int, ECGRecordTicket
        batch_length = self.segment_length * self.batch_size
        signal = wfdb.rdrecord(os.path.splitext(record_ticket.hea_file)[0]).p_signal[
                 local_batch_index * batch_length:(local_batch_index + 1) * batch_length]
        real_batch_size = int(np.ceil(len(signal) / self.segment_length))
        batch_x = [signal[b * self.segment_length:(b + 1) * self.segment_length] for b in range(real_batch_size - 1)]
        batch_x.append(signal[(real_batch_size - 2) * self.segment_length:(real_batch_size - 1) * self.segment_length])
        return np.array(batch_x), np.array([record_ticket.label for _ in range(real_batch_size)])
