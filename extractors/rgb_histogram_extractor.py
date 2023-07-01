import numpy as np
from .extractor import Extractor

class RGBHistogramExtractor(Extractor):
    """
    Extractor that computes RGB histogram from the input images.
    """

    def __init__(self, bins: int) -> None:
        """Constructor method

        :param int bins: Number of bins of the output histograms per colour channel. Should satisfy 1 <= bins <= 256.
        """
        super().__init__(bins=bins)
        self.bins = bins

    def __call__(self, image_paths: list) -> np.ndarray:
        from skimage import io
        from skimage import color
        
        features = []
        for img_path in image_paths:
            rgb = io.imread(img_path)
            if len(rgb.shape) == 2:
                rgb = color.gray2rgb(rgb)
            elif len(rgb.shape) == 4:
                rgb = color.rgba2rgb(rgb)

            feats = []
            for i in range(3):
                histogram, bin_edges = np.histogram(rgb[:,:,i], bins=self.bins, range=(0,256))
                feats.append(histogram)
            
            feats = np.concatenate(feats)
            feats = feats / np.linalg.norm(feats)

            features.append(feats)
        return np.stack(features)
