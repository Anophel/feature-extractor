import numpy as np
from .extractor import Extractor

class VLADExctractor(Extractor):
    def __init__(self, dictionary_path: str = "models/VLAD_dict_64.npy", alpha: int = 0.5) -> None:
        super().__init__()
        import cv2 as cv
        cv.setNumThreads(4)
        self.sift = cv.SIFT_create()
        self.dict = np.load(dictionary_path)
        self.alpha = alpha
        self.epsilon = 1e-14

    def __call__(self, image_paths: list) -> np.ndarray:
        import cv2 as cv
        cv.setNumThreads(4)

        features = []
        for img_path in image_paths:
            img = cv.imread(img_path)
            gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # SIFT detection
            _, des = self.sift.detectAndCompute(gray,None)

            if des is None:
                features.append((np.zeros_like(self.dict) + self.epsilon).flatten())
                continue
            
            # Assigment to the clusters
            assign = np.argmin(np.sqrt(np.sum((des[:,np.newaxis,:] - self.dict)**2, axis=-1)), axis=-1)
            
            # Computing sum of differences for the assigments
            diffs = des - self.dict[assign]
            feat = np.zeros_like(self.dict)
            for i in range(self.dict.shape[0]):
                feat[i] = np.sum(diffs[assign == i])

            feat = feat.flatten()

            # First normalization = "power law normalization"
            feat = np.power(np.abs(feat), self.alpha) * np.sign(feat)

            # Second normalization
            feat = feat / np.linalg.norm(feat)

            features.append(feat)
        return np.stack(features)
