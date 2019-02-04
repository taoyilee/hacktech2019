from core.dataset import LtstdbHea
import pytest
import numpy as np


class TestLtstdbHea(object):
    def test_datatype_from_file(self):
        dataset = LtstdbHea.from_file("test/test_cases/test_001.hea")
        assert type(dataset) == LtstdbHea

    @pytest.mark.parametrize("test_file,sex",
                             [("test/test_cases/test_001.hea", True),
                              ("test/test_cases/test_002.hea", False)])
    def test_datatype_property_sex(self, test_file, sex):
        dataset = LtstdbHea.from_file(test_file)
        assert dataset.sex == sex

    @pytest.mark.parametrize("test_file,age",
                             [("test/test_cases/test_001.hea", 58),
                              ("test/test_cases/test_002.hea", 32)])
    def test_datatype_property_age(self, test_file, age):
        dataset = LtstdbHea.from_file(test_file)
        assert dataset.age == age

    @pytest.mark.parametrize("test_file,name",
                             [("test/test_cases/test_001.hea", "s20011"),
                              ("test/test_cases/test_002.hea", "s20011")])
    def test_hea_name(self, test_file, name):
        dataset = LtstdbHea.from_file(test_file)
        assert dataset.name == name

    @pytest.mark.parametrize("test_file,signals",
                             [("test/test_cases/test_001.hea", 2),
                              ("test/test_cases/test_002.hea", 2)])
    def test_hea_signals(self, test_file, signals):
        dataset = LtstdbHea.from_file(test_file)
        assert len(dataset.signal_spec) == signals

    @pytest.mark.parametrize("test_file,sig_name",
                             [("test/test_cases/test_001.hea", "s20011"),
                              ("test/test_cases/test_002.hea", "s20011")])
    def test_hea_signal_dat_file(self, test_file, sig_name):
        dataset = LtstdbHea.from_file(test_file)
        assert dataset.signal_spec[0].dat_name == sig_name

    @pytest.mark.parametrize("test_file,descr",
                             [("test/test_cases/test_001.hea", ["ML2", "MV2"]),
                              ("test/test_cases/test_002.hea", ["ML2", "MV2"])])
    def test_hea_signal_descr(self, test_file, descr):
        dataset = LtstdbHea.from_file(test_file)
        for s, d in zip(dataset.signal_spec, descr):
            assert s.signal_description == d

    @pytest.mark.parametrize("test_file",
                             ["test/test_cases/test_001.hea",
                              "test/test_cases/test_002.hea",
                              "test/test_cases/test_005.hea"])
    def test_hea_signal_len(self, test_file):
        dataset = LtstdbHea.from_file(test_file)
        for s in dataset.signals:
            assert len(s) == dataset.number_of_samples_per_signal

    @pytest.mark.parametrize("test_file,sampling_freq",
                             [("test/test_cases/test_001.hea", 250),
                              ("test/test_cases/test_002.hea", 250),
                              ("test/test_cases/test_005.hea", 360)])
    def test_hea_sampling_freq(self, test_file, sampling_freq):
        dataset = LtstdbHea.from_file(test_file)
        assert dataset.sampling_freq == sampling_freq

    @pytest.mark.parametrize("test_file,adc_gain",
                             [("test/test_cases/test_001.hea", [200, 200]),
                              ("test/test_cases/test_002.hea", [200, 200])])
    def test_hea_signal_adc_gain(self, test_file, adc_gain):
        dataset = LtstdbHea.from_file(test_file)
        for s, d in zip(dataset.signal_spec, adc_gain):
            assert s.adc_gain == d

    @pytest.mark.parametrize("test_file,adc_checksum",
                             [("test/test_cases/test_001.hea", [-18395, 5078]),
                              ("test/test_cases/test_002.hea", [-18395, 5078])])
    def test_hea_signal_adc_checksum(self, test_file, adc_checksum):
        dataset = LtstdbHea.from_file(test_file)
        for s, d in zip(dataset.signal_spec, adc_checksum):
            assert s.adc_checksum == d

    @pytest.mark.parametrize("test_file,signal_ref",
                             [("test/test_cases/test_001.hea", [[55, 55, 55, 56, 57], [81, 84, 87, 89, 91]])])
    def test_hea_signal(self, test_file, signal_ref):
        dataset = LtstdbHea.from_file(test_file)
        for s, r in zip(dataset.signals, signal_ref):
            assert np.array_equal(s[:len(r)], r)
