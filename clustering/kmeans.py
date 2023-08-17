import numpy as np

from clustering.base import BaseKMeans


class RandomKMeans(BaseKMeans):
    def initialize(self, x: np.ndarray) -> np.ndarray:
        return x[np.random.randint(low=0, high=x.shape[0], size=self.n_clusters)]

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.number:
        return np.linalg.norm(x - y)


class KMeansPlusPlus(BaseKMeans):
    def __first_center(self, x: np.ndarray) -> np.ndarray:
        return x[np.random.randint(0, x.shape[0])]

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.number:
        return np.linalg.norm(x - y)

    def initialize(self, x: np.ndarray) -> np.ndarray:
        centers = []
        centers.append(self.__first_center(x=x))

        for _ in range(1, self.n_clusters):
            probabilities = []

            for point in x:
                distances = []
                for center in centers:
                    distances.append(self.distance(x=center, y=point))
                probabilities.append(np.min(distances))
            probabilities = np.array(probabilities)
            probabilities /= sum(probabilities)

            choice = np.random.choice(a=range(x.shape[0]), p=probabilities)

            centers.append(x[choice])

        return centers
