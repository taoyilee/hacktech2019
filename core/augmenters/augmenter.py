import abc
import numpy as np

class Augmenter(abc.ABC):
    @abc.abstractmethod
    def augment(self, batch: np.ndarray) -> np.ndarray:
        psss
