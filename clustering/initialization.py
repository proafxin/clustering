from collections.abc import Iterable

import numpy as np

from clustering.base import BaseInitialization


class RandomInitialization(BaseInitialization):
    def initial_centers(self, x: Iterable[Iterable] | np.ndarray) -> Iterable[Iterable]:
        return x[np.random.randint(low=0, high=x.shape[0], size=self.n_clusters)]


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
            probabilities = np.array(probabilities)
            probabilities /= sum(probabilities)

            choice = np.random.choice(a=range(x.shape[0]), p=probabilities)

            centers.append(x[choice])

        return centers

    def __first_center(self, x: np.ndarray) -> np.ndarray:
        return x[np.random.randint(0, x.shape[0])]
