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
from typing import Callable


class Dataset:

    def __init__(self, imagelist_path: str, features_path: str, media_server: str = None, media_server_auth: HTTPBasicAuth = None) -> None:
        assert os.path.isfile(
            imagelist_path), "Incorrect imagelist_path. The file does not exist!"
        assert os.path.isfile(
            features_path), "Incorrect features_path. The file does not exist!"

        self._media_server = media_server
        self._media_server_auth = media_server_auth
        with open(imagelist_path, "r") as f:
            self._image_list = [s.strip() for s in f.readlines()]
        with open(features_path, "rb") as f:
            self._features = np.load(f)
            self._features /= np.linalg.norm(self._features,
                                             axis=-1, keepdims=True)
            for i in range(min(5, self._features.shape[0])):
                self_sim = self.get_similarity(i, i)
                assert self_sim > 0.9 and self_sim < 1.1, "Self similarity should be 1.0"
        assert len(
            self._image_list) == self._features.shape[0], "Different size of imagelist and features! Check if they match!!"

    def get_similarity(self, a: int, b: int):
        return np.dot(self._features[a], self._features[b])

    def get_knn(self, target: int, k: int = 4):
        distances = np.dot(self._features, self._features[target])
        return np.argsort(distances)[::-1][:k]

    def get_knn_external(self, target: np.ndarray, k: int = 4):
        assert len(target.shape) == 1, "Target has to be vector"
        assert target.shape[0] == self._features.shape[0], "Target and dataset features has to have the same dimension"

        distances = np.dot(self._features, target)
        return np.argsort(distances)[::-1][:k]

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

    def show_knn(self, target: int, k: int = 4):
        neighbors = self.get_knn(target, k)
        self.show_grid(neighbors, lambda neigh,
                       offset: f"ID: {neigh}, SIM: {self.get_similarity(target, neigh)}")

    def get_nth_neighbours(self, target: int, nths: Iterable):
        distances = np.dot(self._features, self._features[target])
        return np.argsort(distances)[::-1][nths]

    def show_nth_neighbours(self, target: int, nths: Iterable):
        neighbors = self.get_nth_neighbours(target, nths)
        self.show_grid(neighbors, lambda neigh,
                       offset: f"ID: {neigh}, SIM: {self.get_similarity(target, neigh)}, nth: {nths[offset]}")

    def show_knn_external(self, target: np.ndarray, k: int = 4):
        assert len(target.shape) == 1, "Target has to be vector"
        assert target.shape[0] == self._features.shape[0], "Target and dataset features has to have the same dimension"

        neighbors = self.get_knn_external(target, k)
        self.show_grid(neighbors, lambda neigh,
                       offset: f"ID: {neigh}, SIM: {self.get_similarity(target, neigh)}")


