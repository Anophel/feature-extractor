import abc
import numpy as np

class Extractor:
    @abc.abstractmethod
    def __call__(self, image_paths: list) -> np.ndarray:
        pass
