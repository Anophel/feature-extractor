import numpy as np
from .extractor import Extractor

class CIELABPositionalExctractor(Extractor):
    """
    Extractor that converts the image pixels into CIELAB colour space
    and then computes an aggregated pixel value for each sector of an image.
    The aggregation can be computed as a mean, a medoid, or an approximative medoid. 
    """

    def __init__(self, regions: tuple = (4,4), aggType: str = "mean", approx_sample: int = 100) -> None:
        """Constructor method

        :param tuple regions: Number of regions that should be used. (rows, columns)
        :param str aggType: Region aggregation type. Possible values: \"mean\", \"medoid\", \"medoid-approx\".
        :param int approx_sample: Sample size for approximative medoid. Used only when ``aggType == \"medoid-approx\"``.
        """

        super().__init__(regions=regions, aggType=aggType, approx_sample=approx_sample)
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
        """Computed medoid for the region pixel values.

        :param np.ndarray region: Region pixel values

        :returns: Medoid for the region.

        :rtype: np.ndarray
        """

        # Compute distance matrix
        dist_mat = np.linalg.norm(region.reshape((-1,1,3)) - region.reshape((1,-1,3)), axis=-1)
        # Return color with lowest sum of distances
        return region.reshape((-1,3))[np.argmin(np.sum(dist_mat, axis=1))]

    def compute_approx_medoid(self, region: np.ndarray) -> np.ndarray:
        """Computed approximative medoid for the region pixel values.
        The region is subsampled and then the medoid is computed.

        :param np.ndarray region: Region pixel values

        :returns: Approximative medoid for the region.

        :rtype: np.ndarray
        """

        region = region.reshape((-1,3))
        np.random.shuffle(region)
        # Compute distance matrix
        dist_mat = np.linalg.norm(region.reshape((-1,1,3)) - region[:self.approx_sample].reshape((1,-1,3)), axis=-1)
        # Return color with lowest sum of distances
        return region[np.argmin(np.sum(dist_mat, axis=1))]

    def compute_mean(self, region: np.ndarray) -> np.ndarray:
        """Computed mean for the region pixel values.

        :param np.ndarray region: Region pixel values

        :returns: Mean for the region.

        :rtype: np.ndarray
        """
        return np.mean(region, axis=(0,1))

    def __call__(self, image_paths: list) -> np.ndarray:
        from skimage import io
        from skimage import color
        from skimage import transform
        features = []

        for img_path in image_paths:
            rgb = io.imread(img_path)
            if len(rgb.shape) == 2:
                rgb = color.gray2rgb(rgb)
            elif rgb.shape[2] == 4:
                rgb = color.rgba2rgb(rgb)
            
            rgb = transform.resize(rgb, (320, 640))
            
            lab = color.rgb2lab(rgb)
            regions = [arr2 for arr in np.vsplit(lab, self.regions[0]) for arr2 in np.hsplit(arr, self.regions[1]) ]
            features.append(np.array(list(map(self.aggFunction, regions))).flatten())

        return np.stack(features)


