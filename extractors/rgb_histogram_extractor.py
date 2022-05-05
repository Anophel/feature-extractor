import numpy as np
from .extractor import Extractor

class RGBHistogramExtractor(Extractor):
    def __init__(self, bins: int) -> None:
        super().__init__(bins=bins)
        self.bins = bins

    def __call__(self, image_paths: list) -> np.ndarray:
        from skimage import io
        
        features = []
        for img_path in image_paths:
            rgb = io.imread(img_path)

            feats = []
            for i in range(3):
                histogram, bin_edges = np.histogram(rgb[:,:,i], bins=self.bins, range=(0,256))
                feats.append(histogram)
            
            feats = np.concatenate(feats)
            feats = feats / np.linalg.norm(feats)

            features.append(feats)
        return np.stack(features)
