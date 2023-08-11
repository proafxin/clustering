from collections.abc import Iterable

import numpy as np

from clustering.base import BaseInitialization, BaseKMeans
from clustering.distance import EuclideanDistance
from clustering.initialization import RandomInitialization


class KMeans(BaseKMeans):
    def __init__(
        self,
        n_clusters: int | None = None,
        initializer: BaseInitialization | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            initializer=initializer,
        )

        self.__euclidean = EuclideanDistance()

        if not self.initializer:
            self.initializer = RandomInitialization(n_clusters=self.n_clusters)

        if not self.initializer.distance_metric:
            self.initializer.distance_metric = EuclideanDistance()

    def __to_numpy(self, x: Iterable):
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        return x

    def __assign_clusters(self, x: Iterable[Iterable], centers: Iterable[Iterable]):
        x = self.__to_numpy(x=x)

        labels: dict[int, list] = {}
        inertia: float = 0

        for i, point in enumerate(x):
            distances = np.zeros(shape=(len(centers)))
            for j, center in enumerate(centers):
                distances[j] = self.initializer.distance_metric(x=point, y=center)

            label = np.argmin(distances)
            if label not in labels:
                labels[label] = []
            labels[label].append(i)
            inertia += distances[label] ** 2

        return labels, inertia

    def __update_centers(self, x: Iterable[Iterable], labels: dict[int, int]):
        x = self.__to_numpy(x=x)

        centers = np.zeros(shape=(len(labels.keys()), x.shape[1]))
        for label, point_indices in labels.items():
            points = x[point_indices]
            centroid = np.mean(points, axis=0)
            centers[label] = centroid

        return centers

    def fit(self, x: Iterable[Iterable]):
        x = self.__to_numpy(x=x)

        centers = self.initializer.initial_centers(x=x)

        step = 0

        self.inertia_ = []

        while True:
            centers_old = centers.copy()
            labels, inertia = self.__assign_clusters(x=x, centers=centers)
            centers = self.__update_centers(x=x, labels=labels)

            self.inertia_.append(inertia)
            step += 1

            if np.array_equal(centers, centers_old):
                break

        self.iterations_ = step
        self.centers_ = centers

        return self

    def predict(self, x: Iterable[Iterable]) -> Iterable:
        x = self.__to_numpy(x=x)

        labels, _ = self.__assign_clusters(x=x, centers=self.centers_)

        labels_plain = np.zeros(shape=(x.shape[0],))
        for label, point_indices in labels.items():
            labels_plain[point_indices] = label

        return labels_plain

    def fit_predict(self, x: Iterable[Iterable]):
        x = self.__to_numpy(x=x)

        self.fit(x=x)

        return self.predict(x=x)
