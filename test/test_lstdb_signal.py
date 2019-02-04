from core.signal import LtstdbSignal
import pytest


class TestLtstdbSignal(object):
    @pytest.mark.parametrize("test_file,n_signals",
                             [("test/test_cases/s20011.dat", 2)])
    def test_signal_number(self, test_file, n_signals):
        assert len(LtstdbSignal.from_file(test_file, num_signals=n_signals, format=212)) == n_signals

    @pytest.mark.parametrize("test_file,signal_len", [("test/test_cases/s20011.dat", 20594750)])
    def test_signal_length(self, test_file, signal_len):
        assert len(LtstdbSignal.from_file(test_file, num_signals=2, format=212)[0]) == signal_len
        assert len(LtstdbSignal.from_file(test_file, num_signals=2, format=212)[1]) == signal_len
