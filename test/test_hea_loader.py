from core.dataset.hea_loader import HeaLoader, HeaLoaderFixedLabel, HeaLoaderExcel
import pytest
import configparser as cp

config = cp.ConfigParser()
config.read("config.ini")
mitdb_path, nsrdb_path = config["mitdb"].get("dataset_path"), config["nsrdb"].get("dataset_path")


class TestHeaLoader(object):
    @classmethod
    def setup_class(cls):
        cls.hea_loader = HeaLoader

    @pytest.mark.parametrize("hea_directory, label",
                             [(mitdb_path, "mitdb_labeled.xlsx"), (nsrdb_path, 0), (nsrdb_path, 1)])
    def test_no_error(self, hea_directory, label):
        try:
            self.hea_loader.load(hea_directory, label)
        except Exception as e:
            pytest.fail(f"{e}")

    @pytest.mark.parametrize("hea_directory, label, datatype",
                             [(mitdb_path, "mitdb_labeled.xlsx", HeaLoaderExcel), (nsrdb_path, 0, HeaLoaderFixedLabel)])
    def test_correct_type(self, hea_directory, label, datatype):
        assert isinstance(self.hea_loader.load(hea_directory, label), datatype)

    @pytest.mark.parametrize("hea_directory, label, datatype",
                             [(mitdb_path, "mitdb_labeled.xlsx", HeaLoaderFixedLabel), (nsrdb_path, 0, HeaLoaderExcel)])
    def test_correct_type_xfail(self, hea_directory, label, datatype):
        assert not isinstance(self.hea_loader.load(hea_directory, label), datatype)

    @pytest.mark.parametrize("record, start_idx, ending_idx, expected_label",
                             [(100, 0, 3600, True),
                              (100, 3601, 7200, True),
                             (100, 21600, 25200, False),
                              (100, 21500, 25100, True)])
    def test_get_label(self, record, start_idx, ending_idx, expected_label):
        instance = self.hea_loader.load(mitdb_path, "mitdb_labeled.xlsx")
        assert instance.get_label(record, start_idx, ending_idx) == expected_label
