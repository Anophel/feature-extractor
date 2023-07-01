import numpy as np
from .extractor import Extractor

class CIELABKMeansExctractor(Extractor):
    """
    Extractor that converts the image pixels into CIELAB colour space
    and then computes KMean on these values. The k parameter for KMeans
    is selected in constructor. Final feature vector are concatenated 
    CIELAB values from the KMeans centroids ordered by their hue. 
    """

    def __init__(self, k: int) -> None:
        """Constructor method

        :param int k: Parameter k for KMeans.
        """
        super().__init__(k=k)
        self.k = k

    def __call__(self, image_paths: list) -> np.ndarray:
        from skimage import io
        from skimage import color
        from sklearn.cluster import KMeans
        
        features = []
        for img_path in image_paths:
            rgb = io.imread(img_path)
            if len(rgb.shape) == 2:
                rgb = color.gray2rgb(rgb)
            elif rgb.shape[2] == 4:
                rgb = color.rgba2rgb(rgb)
            
            lab = color.rgb2lab(rgb).reshape((-1, 3))
            feats = KMeans(n_clusters=self.k).fit(lab).cluster_centers_
            feats_hsv = color.rgb2hsv(color.lab2rgb(feats))
            features.append(feats[np.argsort(feats_hsv[:,0])].flatten())
        return np.stack(features)
