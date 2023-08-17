from abc import ABC, abstractmethod

import numpy as np


class BaseKMeans(ABC):
    def __init__(self, n_clusters: int) -> None:
        self.n_clusters = n_clusters

        self.iterations_ = 0
        self.inertia_: list = []
        self.centers_: np.ndarray

    @abstractmethod
    def assign_clusters(self, x: np.ndarray, centers: np.ndarray):
        pass

    @abstractmethod
    def update_centers(self, x: np.ndarray, labels: dict[int, int]):
        pass

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> np.number:
        pass

    @abstractmethod
    def initialize(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit(self, x: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def fit_predict(self, x: np.ndarray):
        self.fit(x=x)

        return self.predict(x=x)
