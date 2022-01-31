import numpy as np

class BatchExtractor:

    def __init__(self, batch_size : int, extractor) -> None:
        self.batch_size = batch_size
        self.extractor = extractor

    def __call__(self, image_paths: list) -> np.ndarray:
        results = []
        for i in range(0, len(image_paths), self.batch_size):
            results.append(self.extractor(image_paths[i:i + self.batch_size]))
        return np.concatenate(results)

    

