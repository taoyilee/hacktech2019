from core.dataset import Format212reader
import pytest
from bitarray import bitarray
import numpy as np


class TestLtstdbHea(object):
    @classmethod
    def setup_class(cls):
        cls.reader = Format212reader()

    @pytest.mark.parametrize("bytes_array,array",
                             [(b"\x01\x00\x00\x01\x00\x00", [1, 0, 1, 0]),
                              (bitarray("100000000000100000000000", endian="little").tobytes(), [1, 256])])
    def test_from_bytes(self, bytes_array, array):
        signal = self.reader.from_bytes(bytes_array)
        assert np.array_equal(signal, array)

    @pytest.mark.parametrize("test_file,length",
                             [("test/test_cases/short.dat", 62160), ("test/test_cases/s20011.dat", 20594750 * 2)])
    def test_signal_length(self, test_file, length):
        with open(test_file, "rb") as f:
            signal = self.reader.from_bytestream(f)
        assert len(signal) == length

    @pytest.mark.parametrize("read_size", [3 * 1024, 3 * 2048, 3 * 4096])
    def test_signal_readsize(self, read_size):
        with open("test/test_cases/short.dat", "rb") as f:
            signal = self.reader.from_bytestream(f, read_size)
        assert len(signal) == 62160

    @pytest.mark.parametrize("test_file,array",
                             [("test/test_cases/short.dat", [55, 81, 55, 81, 55, 81, 55, 81, 55, 81])])
    def test_signal_begin(self, test_file, array):
        with open(test_file, "rb") as f:
            signal = self.reader.from_bytestream(f)
        assert np.array_equal(signal[:len(array)], array)

    @pytest.mark.parametrize("test_file,array",
                             [("test/test_cases/s20011.dat", [-8, 14, -9, 14, -10, 14, -10, 14, -10, 14])])
    def test_signal_end(self, test_file, array):
        with open(test_file, "rb") as f:
            signal = self.reader.from_bytestream(f)
        assert np.array_equal(signal[-len(array):], array)
