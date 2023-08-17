import pandas as pd

from clustering.dataframe_kmeans import ClusterDataframe
from clustering.kmeans import KMeansPlusPlus, RandomKMeans


df = pd.read_csv("data/iris.csv")
n_clusters = 3

kmeans = {
    "random": RandomKMeans(n_clusters=n_clusters),
    "auto": KMeansPlusPlus(n_clusters=n_clusters),
}

for mode in kmeans:
    clusterer = ClusterDataframe(kmeans=kmeans[mode])
    labels = clusterer.cluster(df=df.iloc[:, :-1])

    print(mode)
    print(labels)
    print(clusterer.kmeans.inertia_, clusterer.kmeans.iterations_)
