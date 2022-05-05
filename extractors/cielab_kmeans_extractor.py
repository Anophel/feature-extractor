import numpy as np
from .extractor import Extractor

class CIELABKMeansExctractor(Extractor):
    def __init__(self, k: int) -> None:
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
            elif len(rgb.shape) == 4:
                rgb = color.rgba2rgb(rgb)
            
            lab = color.rgb2lab(rgb).reshape((-1, 3))
            feats = KMeans(n_clusters=self.k).fit(lab).cluster_centers_
            feats_hsv = color.rgb2hsv(color.lab2rgb(feats))
            features.append(feats[np.argsort(feats_hsv[:,0])].flatten())
        return np.stack(features)
