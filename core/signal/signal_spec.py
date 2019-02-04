from lark import Tree
from core.signal.signal_spec_visitor import SignalSpecVisitor


class SignalSpec:
    dat_name = None
    signal_format = None
    adc_gain = None
    adc_units = None
    adc_resol = None
    adc_zero = None
    adc_init_val = None
    adc_checksum = None
    adc_block_size = None
    signal_description = None

    @classmethod
    def from_tree(cls, tree: Tree):
        new_instance = cls()
        SignalSpecVisitor(new_instance).visit(tree)
        return new_instance
