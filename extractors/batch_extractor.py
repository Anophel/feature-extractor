import numpy as np
from .extractor import Extractor
from alive_progress import alive_bar
import math
from typing import Callable


class BatchExtractor(Extractor):
    """
    Helper extractor wrapper to process feature extraction per batches becuase\
    most of the extractors process the given images in one batch.
    """

    def __init__(self, batch_size: int, extractor: Extractor, show_progress: bool = False) -> None:
        """Constructor method

        :param int batch_size: Batch size
        :param Extractor extractor: Inner extractor
        :param bool show_progress: Flag if the progress should be shown
        """
        super().__init__(extractor=extractor)
        self.batch_size = batch_size
        self.extractor = extractor
        self._extract_inner = self.extract_with_progress if show_progress else self.extract

    def extract_with_progress(self, image_paths: list):
        """Inner method for running the extraction with the progress.

        :param list image_paths: Image paths
        """
        with alive_bar(math.ceil(len(image_paths) / self.batch_size)) as bar:
            return self.extract(image_paths, bar)

    def extract(self, image_paths: list, callback: Callable = lambda: True):
        """Inner method for running the extraction with a custom callback.

        :param list image_paths: Image paths
        """
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            results.append(self.extractor(image_paths[i:i + self.batch_size]))
            callback()
        return np.concatenate(results)

    def __call__(self, image_paths: list) -> np.ndarray:
        return self._extract_inner(image_paths)
