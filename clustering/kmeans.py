import numpy as np

from clustering.base import BaseKMeans


class RandomKMeans(BaseKMeans):
    def initialize(self, x: np.ndarray) -> np.ndarray:
        return x[np.random.randint(low=0, high=x.shape[0], size=self.n_clusters)]

    def distance(self, x: np.ndarray, y: np.ndarray) -> np.number:
        return np.linalg.norm(x - y)

    def update_centers(self, x: np.ndarray, labels: dict[int, int]):
        centers = np.zeros(shape=(len(labels.keys()), x.shape[1]))
        for label, point_indices in labels.items():
            points = x[point_indices]
            centroid = np.mean(points, axis=0)
            centers[label] = centroid

        return centers

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

    def update_centers(self, x: np.ndarray, labels: dict[int, int]):
        centers = np.zeros(shape=(len(labels.keys()), x.shape[1]))
        for label, point_indices in labels.items():
            points = x[point_indices]
            centroid = np.mean(points, axis=0)
            centers[label] = centroid

        return centers

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
