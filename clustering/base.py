from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Number

import numpy as np


class DistanceMetric(ABC):
    @abstractmethod
    def distance(self, x: Number | np.number, y: Number | np.number):
        pass

    def __call__(
        self, x: Number | np.number, y: Number | np.number
    ) -> Number | np.number:
        return self.distance(x, y)


class BaseInitialization(ABC):
    def __init__(
        self, n_clusters: int, distance_metric: DistanceMetric | None = None
    ) -> None:
        super().__init__()

        self.n_clusters = n_clusters
        self.distance_metric = distance_metric

    @abstractmethod
    def initial_centers(self, x: Iterable[Iterable] | np.ndarray) -> Iterable[Iterable]:
        pass


class BaseKMeans(ABC):
    def __init__(
        self,
        n_clusters: int | None,
        initializer: BaseInitialization | None = None,
    ) -> None:
        self.initializer = None

        if initializer:
            if n_clusters and n_clusters != initializer.n_clusters:
                raise ValueError(
                    f"`n_clusters`: {n_clusters} != `initializer.n_clusters`: {initializer.n_clusters}"
                )

            self.initializer = initializer
            self.n_clusters = initializer.n_clusters
        else:
            if not n_clusters:
                raise ValueError("Both `n_clusters` and `initializer` cannot be null.")

            self.n_clusters = n_clusters

        self.iterations_ = 0
        self.inertia_ = 0
        self.centers_ = []

    @abstractmethod
    def fit(self, x: np.ndarray | Iterable[Iterable]):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray | Iterable[Iterable]) -> Iterable:
        pass

    @abstractmethod
    def fit_predict(self, x: np.ndarray | Iterable[Iterable]) -> Iterable:
        pass

    def __str__(self) -> str:
        return f"Kmeans {self.n_clusters}, {self.initializer.__class__}"
