import abc
import numpy as np

class Extractor:
    """
    Base class for all the extractors. Defines the interface for the image feature extraction\
    and human readable name of the extractor.
    """

    def __init__(self, **kwargs) -> None:
        self.__name__ = f"{type(self).__name__}({kwargs})" 

    @abc.abstractmethod
    def __call__(self, image_paths: list) -> np.ndarray:
        """Extraction method. It takes list of image paths and return feature matrix as a numpy array. 

        :param list image_paths: Image paths

        :returns: Feature matrix

        :rtype: np.ndarray
        """
        pass

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__