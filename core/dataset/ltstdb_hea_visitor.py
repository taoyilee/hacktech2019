from lark import Tree, Visitor
from core.signal.signal_spec import SignalSpec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.dataset import LtstdbHea


class LtstdbHeaVisitor(Visitor):
    def __init__(self, new_instance: "LtstdbHea"):
        self.new_instance = new_instance

    def record_name(self, tree: Tree):
        assert tree.data == "record_name"
        self.new_instance.name = tree.children[0]

    def signal_spec(self, tree: Tree):
        assert tree.data == "signal_spec"
        self.new_instance.signal_spec.append(SignalSpec.from_tree(tree))

    def sampling_frequency(self, tree: Tree):
        assert tree.data == "sampling_frequency"
        self.new_instance.sampling_freq = tree.children[0]

    def number_of_samples_per_signal(self, tree: Tree):
        assert tree.data == "number_of_samples_per_signal"
        self.new_instance.number_of_samples_per_signal = tree.children[0]
