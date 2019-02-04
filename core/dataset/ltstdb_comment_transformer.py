from lark import Transformer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.dataset import LtstdbHea


# TODO: process comments / diagnosis / treatments ... etc and write into property of new_instance
class LtstdbCommentTransformer(Transformer):
    def __init__(self, new_instance: "LtstdbHea"):
        self.new_instance = new_instance

    def sex_options(self, args):
        if args[0] == "F":
            self.new_instance.sex = False
        else:
            self.new_instance.sex = True

    def age(self, args):
        self.new_instance.age = args[0]

    def positive(self, args):
        return True

    def negative(self, args):
        return False

    def integer(self, args):
        return int(args[0])
