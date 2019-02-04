from core.dataset import LtstdbCommentParser

import pytest


class TestLtstdbCommentParser(object):
    @classmethod
    def setup_class(cls):
        cls.parser = LtstdbCommentParser()

    @pytest.mark.parametrize("test_file",
                             ["test/test_cases/comment_tc_001.txt",
                              "test/test_cases/comment_tc_002.txt",
                              "test/test_cases/comment_tc_003.txt"])
    def test_no_error(self, test_file):
        try:
            self.parser.parse_file(test_file)
        except Exception as e:
            pytest.fail(f"{e}")
