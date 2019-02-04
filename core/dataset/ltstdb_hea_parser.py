from lark import Lark, Tree
import os
from typing import List


class LtstdbHeaParser:
    def __init__(self):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "grammar/hea.g"), "r") as f:
            grammar = f.read()
        self.parser = Lark(grammar, parser="lalr")

    def parse_file(self, hea_file):
        with open(hea_file, "r") as f:
            lines = f.read()
        return self.parse(lines)

    def parse(self, lines):
        tree = self.parser.parse(lines)
        return tree

    def extract_comments(self, hea_file):
        tree = self.parse_file(hea_file)
        comments = list(tree.find_data("comment"))  # type: List[Tree]
        return "\n".join([c.children[0].strip("#") for c in comments])
