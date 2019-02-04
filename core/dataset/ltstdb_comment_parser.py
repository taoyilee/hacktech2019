from lark import Lark, Tree
import os
from typing import List


class LtstdbCommentParser:
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "grammar/ltstdb_comment.g"), "r") as f:
            grammar = f.read()
        self.parser = Lark(grammar, parser="earley")

    def parse_file(self, hea_file):
        with open(hea_file, "r") as f:
            lines = f.read()
        return self.parse(lines)

    def parse(self, lines):
        tree = self.parser.parse(lines)
        return tree
