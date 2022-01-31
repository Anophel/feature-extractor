import numpy as np
from skimage import io
from skimage import color
from sklearn.cluster import KMeans

class CIELABKMeansExctractor:
    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, image_paths: list) -> np.ndarray:
        features = []
        for img_path in image_paths:
            rgb = io.imread(img_path)
            lab = color.rgb2lab(rgb).reshape((-1, 3))
            feats = KMeans(n_clusters=self.k).fit(lab).cluster_centers_
            norms = np.linalg.norm(feats, axis=1)
            features.append(feats[np.argsort(norms)].flatten())
        return np.stack(features)
