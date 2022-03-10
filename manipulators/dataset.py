from io import BytesIO
import math
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from collections.abc import Iterable
from requests.auth import HTTPBasicAuth
import requests
from typing import Callable, Union


class Dataset:

    def __init__(self, imagelist_path: str, features_path: str, media_server: str = None, media_server_auth: HTTPBasicAuth = None) -> None:
        assert os.path.isfile(
            imagelist_path), "Incorrect imagelist_path. The file does not exist!"
        assert os.path.isfile(
            features_path), "Incorrect features_path. The file does not exist!"

        self._media_server = media_server
        self._media_server_auth = media_server_auth
        with open(imagelist_path, "r") as f:
            self._image_list = np.array([s.strip() for s in f.readlines()])
        with open(features_path, "rb") as f:
            self._features = np.load(f)
            self._features /= np.linalg.norm(self._features,
                                             axis=-1, keepdims=True)
            for i in range(min(5, self._features.shape[0])):
                self_sim = self.get_similarity(i, i)
                assert self_sim > 0.9 and self_sim < 1.1, "Self similarity should be 1.0"
        assert len(
            self._image_list) == self._features.shape[0], "Different size of imagelist and features! Check if they match!!"

    def get_image(self, target: int) -> BytesIO:
        if self._media_server is None:
            with open(self._image_list[target], 'rb') as f:
                return BytesIO(f.read())
        else:
            response = requests.get(
                self._media_server + "/" + self._image_list[target], auth=HTTPBasicAuth('som', 'hunter'))
            return BytesIO(response.content)

    def show(self, target: int, ax: Axes = None):
        image = io.imread(self.get_image(target))
        if ax is None:
            plt.imshow(image)
            plt.axis('off')
            plt.show()
        else:
            ax.imshow(image)
            ax.set_axis_off()

    def show_grid(self, ids: Iterable, title_callback: Callable[[int, int], str] = lambda x, y: f"{x}"):
        k = len(ids)
        fig, axes = plt.subplots(ncols=4, nrows=int(
            math.ceil(k / 4)), figsize=(20, k * 6 // 10))

        if k <= 4:
            axes = [axes]

        for ax, neigh, offset in zip([item for sublist in axes for item in sublist], ids, range(k)):
            ax.set_title(title_callback(neigh, offset))
            self.show(neigh, ax=ax)
        
        for ax in [item for sublist in axes for item in sublist][offset:]:
            ax.set_axis_off()

        plt.tight_layout()
        plt.show()

    def _get_feature_vector(self, target: Union[int, np.ndarray]) -> np.ndarray:
        if type(target) is np.ndarray:
            target /= np.linalg.norm(target, axis=-1, keepdims=True)
            assert len(target.shape) == 1, "Target has to be vector"
            assert target.shape[0] == self._features.shape[1], "Target and dataset features has to have the same dimension"
        else:
            target = self._features[target]
        return target

    def get_similarity(self, target: Union[int, np.ndarray], other: int):
        target = self._get_feature_vector(target)
        return np.dot(target, self._features[other])

    def get_knn(self, target: Union[int, np.ndarray], k: int = 4):
        target = self._get_feature_vector(target)
        distances = np.dot(self._features, target)
        return np.argsort(distances)[::-1][:k]

    def show_knn(self, target: Union[int, np.ndarray], k: int = 4):
        neighbors = self.get_knn(target, k)
        self.show_grid(neighbors, lambda neigh,
                       offset: f"ID: {neigh}, SIM: {self.get_similarity(target, neigh)}")

    def get_nth_neighbours(self, target: Union[int, np.ndarray], nths: Iterable):
        target = self._get_feature_vector(target)
        distances = np.dot(self._features, target)
        return np.argsort(distances)[::-1][nths]

    def show_nth_neighbours(self, target: Union[int, np.ndarray], nths: Iterable):
        neighbors = self.get_nth_neighbours(target, nths)
        self.show_grid(neighbors, lambda neigh,
                       offset: f"ID: {neigh}, SIM: {self.get_similarity(target, neigh)}, nth: {nths[offset]}")

    def filter_knn(self, target: Union[int, np.ndarray], k: int):
        neighbors = self.get_knn(target, k)
        mask = np.ones(self._features.shape[0], dtype=bool)
        mask[neighbors] = False
        self._features = self._features[mask]
        self._image_list = self._image_list[mask]

