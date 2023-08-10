from collections.abc import Iterable

import numpy as np

from clustering.base import BaseInitialization


class RandomInitialization(BaseInitialization):
    def initial_centers(self, x: Iterable[Iterable] | np.ndarray) -> Iterable[Iterable]:
        return np.random.choice(a=x, size=self.n_clusters)


class KmeansPlusPlusInitialization(BaseInitialization):
    def initial_centers(self, x: Iterable[Iterable] | np.ndarray) -> Iterable[Iterable]:
        if not isinstance(x, np.ndarray):
            x = np.array(x)

        centers = []
        centers.append(self.__first_center(x=x))

        for _ in range(1, self.n_clusters):
            probabilities = []

            for point in x:
                distances = []
                for center in centers:
                    distances.append(self.distance_metric(x=center, y=point))
                probabilities.append(np.min(distances))
            probabilities /= sum(probabilities)

            centers.extend(np.random.choice(a=x, size=1, p=probabilities))

        return centers

    def __first_center(self, x: np.ndarray) -> np.ndarray:
        return np.random.choice(x)
