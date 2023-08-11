from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from clustering.kmeans import KMeans


@dataclass
class ClusterDataframe:
    n_clusters: int | None = None
    kmeans: KMeans | None = None

    def __post_init__(self):
        if not self.n_clusters and not self.kmeans:
            raise ValueError("Both `n_clusters` and `kmeans` are null.")

        if not self.kmeans:
            self.kmeans = KMeans(n_clusters=self.n_clusters)

    def cluster(self, df: pd.DataFrame) -> Iterable[int]:
        x = df.to_numpy()

        return self.kmeans.fit_predict(x=x)
