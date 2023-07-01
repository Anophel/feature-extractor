import numpy as np
import cv2
from extractors.extractor import Extractor

class BlurryExtractor(Extractor):
    """
    Blurry extractor that produce a single float value for each image.
    The graeter the value the less blurry the image is.
    """

    def __call__(self, image_paths: list) -> np.ndarray:
        features = []
        for path in image_paths:
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurr = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append([blurr])
        return np.array(features)
