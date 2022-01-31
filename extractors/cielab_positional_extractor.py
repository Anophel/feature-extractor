import numpy as np
from .extractor import Extractor
from skimage import io
from skimage import color

class CIELABPositionalExctractor(Extractor):
    def __init__(self, regions: tuple = (4,4), aggType: str = "medoid", approx_sample: int = 100) -> None:
        # Regions = (rows, columns)
        self.regions = regions
        # Only for aggType == "medoid-approx"
        self.approx_sample = approx_sample
        self.aggType = aggType.lower()
        if self.aggType == "medoid":
            self.aggFunction = self.compute_medoid
        elif self.aggType == "medoid-approx":
            self.aggFunction = self.compute_approx_medoid
        elif self.aggType == "mean":
            self.aggFunction = self.compute_mean
        else:
            raise Exception("Unknown agg type")

    def compute_medoid(self, region: np.ndarray) -> np.ndarray:
        # Compute distance matrix
        dist_mat = np.linalg.norm(region.reshape((-1,1,3)) - region.reshape((1,-1,3)), axis=-1)
        # Return color with lowest sum of distances
        return region.reshape((-1,3))[np.argmin(np.sum(dist_mat, axis=1))]

    def compute_approx_medoid(self, region: np.ndarray) -> np.ndarray:
        region = region.reshape((-1,3))
        np.random.shuffle(region)
        # Compute distance matrix
        dist_mat = np.linalg.norm(region.reshape((-1,1,3)) - region[:self.approx_sample].reshape((1,-1,3)), axis=-1)
        # Return color with lowest sum of distances
        return region[np.argmin(np.sum(dist_mat, axis=1))]

    def compute_mean(self, region: np.ndarray) -> np.ndarray:
        return np.mean(region, axis=(0,1))

    def __call__(self, image_paths: list) -> np.ndarray:
        features = []

        for img_path in image_paths:
            rgb = io.imread(img_path)
            lab = color.rgb2lab(rgb)
            regions = [arr2 for arr in np.vsplit(lab, self.regions[0]) for arr2 in np.hsplit(arr, self.regions[1]) ]
            features.append(np.array(list(map(self.aggFunction, regions))))

        return np.stack(features)


