from core.dataset import LtstdbHeaParser
import pytest
from core.util.test_helper import list_all_hea


class TestLtstdbHeaParser(object):
    @classmethod
    def setup_class(cls):
        cls.parser = LtstdbHeaParser()

    @pytest.mark.parametrize("test_file",
                             ["test/test_cases/test_001.hea",
                              "test/test_cases/test_002.hea",
                              "test/test_cases/test_003.hea",
                              "test/test_cases/test_004.hea"])
    def test_no_error(self, test_file):
        try:
            self.parser.parse_file(test_file)
        except Exception as e:
            pytest.fail(f"{e}")

    @pytest.mark.parametrize("test_file", list_all_hea())
    def test_scan_all(self, test_file):
        try:
            self.parser.parse_file(test_file)
        except Exception as e:
            pytest.fail(f"{e}")

    @pytest.mark.parametrize("test_file,comment_file",
                             [("test/test_cases/test_001.hea", "test/test_cases/comment_001.txt"),
                              ("test/test_cases/test_002.hea", "test/test_cases/comment_002.txt"),
                              ("test/test_cases/test_003.hea", "test/test_cases/comment_003.txt"),
                              ("test/test_cases/test_004.hea", "test/test_cases/comment_004.txt")])
    def test_extract_comments(self, test_file, comment_file):
        actual = self.parser.extract_comments(test_file)
        actual_lines = actual.split("\n")
        with open(comment_file, "r") as f:
            comment_lines = f.readlines()
        for i, a in enumerate(actual_lines):
            assert a == comment_lines[i].strip("\n")
