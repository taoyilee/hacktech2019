from core.dataset import ECGDataset
import pytest
import configparser as cp


class TestECGDataset(object):
    @classmethod
    def setup_class(cls):
        config = cp.ConfigParser()
        config.read("config.ini")
        cls.mitdb_path, cls.nsrdb_path = config["mitdb"].get("dataset_npy_path"), config["nsrdb"].get(
            "dataset_npy_path")

    def test_length_1(self):
        mitdb = ECGDataset.from_directory(self.mitdb_path, None)
        assert len(mitdb) == 48

    def test_length_2(self):
        nsrdb = ECGDataset.from_directory(self.nsrdb_path, None)
        assert len(nsrdb) == 18

    def test_length_add_1(self):
        mitdb = ECGDataset.from_directory(self.mitdb_path, None)
        nsrdb = ECGDataset.from_directory(self.nsrdb_path, None)
        assert len(mitdb + nsrdb) == (len(mitdb) + len(nsrdb))

    def test_length_add_2(self):
        mitdb = ECGDataset.from_directory(self.mitdb_path, None)[0:3]
        nsrdb = ECGDataset.from_directory(self.nsrdb_path, None)[0:2]
        assert len(mitdb + nsrdb) == (len(mitdb) + len(nsrdb))

    def test_length_add_3(self):
        mitdb = ECGDataset.from_directory(self.mitdb_path, None)[0:2]
        nsrdb = ECGDataset.from_directory(self.nsrdb_path, None)[0:2]
        assert len(mitdb + nsrdb) == (len(mitdb) + len(nsrdb))

    def test_length_add_4(self):
        mitdb = ECGDataset.from_directory(self.mitdb_path, None)[5]
        nsrdb = ECGDataset.from_directory(self.nsrdb_path, None)[4]
        assert len(mitdb + nsrdb) == (len(mitdb) + len(nsrdb))
