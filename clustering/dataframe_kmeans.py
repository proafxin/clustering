from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clustering.base import BaseKMeans


@dataclass
class ClusterDataframe:
    kmeans: BaseKMeans

    def cluster(self, df: pd.DataFrame) -> Iterable[int]:
        x = df.to_numpy()

        return self.kmeans.fit_predict(x=x)

    @property
    def inertia(self) -> np.ndarray:
        return self.kmeans.inertia_

    @property
    def iterations(self) -> np.ndarray:
        return self.kmeans.iterations_
