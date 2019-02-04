from lark import Transformer, v_args
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.dataset import LtstdbHea


class LtstdbHeaTransformer(Transformer):
    def __init__(self, new_instance: "LtstdbHea"):
        self.new_instance = new_instance

    @v_args(inline=True)
    def integer(self, number):
        return int(number)

    @v_args(inline=True)
    def signed_integer(self, number):
        return int(number)

    @v_args(inline=True)
    def signal_description_options(self, opt):
        return opt
