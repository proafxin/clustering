from abc import ABC, abstractmethod

import numpy as np


class BaseKMeans(ABC):
    def __init__(self, n_clusters: int, random_seed: int | None = None) -> None:
        self.n_clusters = n_clusters

        self.iterations_ = 0
        self.inertia_: list = []
        self.centers_: np.ndarray

        if random_seed:
            np.random.seed(seed=random_seed)

    def assign_clusters(self, x: np.ndarray, centers: np.ndarray):
        labels: dict[int, list] = {}
        inertia: float = 0

        np.mean(x, axis=0)

        for i, point in enumerate(x):
            distances = np.zeros(shape=(len(centers)))
            for j, center in enumerate(centers):
                distances[j] = self.distance(x=point, y=center)

            label = np.argmin(distances)
            if label not in labels:
                labels[label] = []
            labels[label].append(i)
            inertia += distances[label] ** 2

        return labels, inertia

    def update_centers(self, x: np.ndarray, labels: dict[int, int]):
        centers = np.zeros(shape=(len(labels.keys()), x.shape[1]))
        for label, point_indices in labels.items():
            points = x[point_indices]
            centroid = np.mean(points, axis=0)
            centers[label] = centroid

        return centers

    @abstractmethod
    def distance(self, x: np.ndarray, y: np.ndarray) -> np.number:
        pass

    @abstractmethod
    def initialize(self, x: np.ndarray) -> np.ndarray:
        pass

    def fit(self, x: np.ndarray):
        centers = self.initialize(x=x)

        step = 0
        self.inertia_ = []

        while True:
            centers_old = centers.copy()
            labels, inertia = self.assign_clusters(x=x, centers=centers)
            centers = self.update_centers(x=x, labels=labels)

            self.inertia_.append(inertia)
            step += 1

            if np.array_equal(centers, centers_old):
                break

        self.iterations_ = step
        self.centers_ = centers

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        labels, _ = self.assign_clusters(x=x, centers=self.centers_)

        labels_plain = np.zeros(shape=(x.shape[0],))
        for label, point_indices in labels.items():
            labels_plain[point_indices] = label

        return labels_plain

    def fit_predict(self, x: np.ndarray):
        self.fit(x=x)

        return self.predict(x=x)
