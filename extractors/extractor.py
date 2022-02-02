import abc
import numpy as np

class Extractor:

    def __init__(self, **kwargs) -> None:
        self.__name__ = f"{type(self).__name__}({kwargs})" 

    @abc.abstractmethod
    def __call__(self, image_paths: list) -> np.ndarray:
        pass

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__