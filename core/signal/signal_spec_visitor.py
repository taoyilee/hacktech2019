from lark import Tree, Visitor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.signal.signal_spec import SignalSpec


class SignalSpecVisitor(Visitor):
    def __init__(self, new_instance: "SignalSpec"):
        self.new_instance = new_instance

    def dat_name(self, tree: Tree):
        assert tree.data == "dat_name"
        self.new_instance.dat_name = tree.children[0]

    def signal_format(self, tree: Tree):
        assert tree.data == "signal_format"
        self.new_instance.signal_format = tree.children[0]

    def adc_gain(self, tree: Tree):
        assert tree.data == "adc_gain"
        self.new_instance.adc_gain = tree.children[0]

    def adc_units(self, tree: Tree):
        assert tree.data == "adc_units"
        self.new_instance.adc_units = tree.children[0]

    def adc_resol(self, tree: Tree):
        assert tree.data == "adc_resol"
        self.new_instance.adc_resol = tree.children[0]

    def adc_zero(self, tree: Tree):
        assert tree.data == "adc_zero"
        self.new_instance.adc_zero = tree.children[0]

    def adc_init_val(self, tree: Tree):
        assert tree.data == "adc_init_val"
        self.new_instance.adc_init_val = tree.children[0]

    def adc_checksum(self, tree: Tree):
        assert tree.data == "adc_checksum"
        self.new_instance.adc_checksum = tree.children[0]

    def adc_block_size(self, tree: Tree):
        assert tree.data == "adc_block_size"
        self.new_instance.adc_block_size = tree.children[0]

    def signal_description(self, tree: Tree):
        assert tree.data == "signal_description"
        self.new_instance.signal_description = tree.children[0]
