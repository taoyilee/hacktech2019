from core.dataset import LtstdbHeaParser
from core.dataset import LtstdbCommentParser

from core.util.test_helper import list_all_hea
import pytest


class TestLtstdbHeaCommentParser(object):
    @classmethod
    def setup_class(cls):
        cls.parser = LtstdbHeaParser()
        cls.cmt_parser = LtstdbCommentParser()

    @pytest.mark.parametrize("test_file", list_all_hea())
    def test_scan_all(self, test_file):
        try:
            self.cmt_parser.parse(self.parser.extract_comments(test_file))
        except Exception as e:
            pytest.fail(f"{e}")
