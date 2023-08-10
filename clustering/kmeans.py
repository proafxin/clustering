from collections.abc import Iterable

import numpy as np

from clustering.base import BaseInitialization, BaseKMeans, DistanceMetric
from clustering.distance import EuclideanDistance
from clustering.initialization import RandomInitialization


class KMeans(BaseKMeans):
    def __init__(
        self,
        n_clusters: int | None = None,
        initializer: BaseInitialization | None = None,
        distance_metric: DistanceMetric | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            initializer=initializer,
            distance_metric=distance_metric,
        )

        if not self.initializer:
            self.initializer = RandomInitialization(n_clusters=self.n_clusters)

        if not self.distance_metric:
            self.distance_metric = EuclideanDistance()

    def __update_centers(
        self,
        x: np.ndarray | Iterable[Iterable],
        centers: Iterable[Iterable] | np.ndarray,
    ):
        pass

    def fit(self, x: np.ndarray | Iterable[Iterable]):
        return self

    def predict(self, x: np.ndarray | Iterable[Iterable]) -> Iterable:
        return []

    def fit_predict(self, x: np.ndarray | Iterable):
        return []
