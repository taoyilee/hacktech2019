from .augmenter import Augmenter

class RandomScaleAugmenter(Augmenter):
    def __init__(self):
        self.seed =