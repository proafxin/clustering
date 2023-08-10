from numbers import Number

import numpy as np

from clustering.base import DistanceMetric


class EuclideanDistance(DistanceMetric):
    def distance(self, x: Number | np.number, y: Number | np.number):
        diff = x - y
        return np.sqrt(np.dot(diff, diff))


class EuclideanDistanceSquared(DistanceMetric):
    def distance(self, x: Number | np.number, y: Number | np.number):
        diff = x - y
        return np.dot(diff, diff)
