import pandas as pd

from clustering.dataframe_kmeans import ClusterDataframe
from clustering.initialization import KmeansPlusPlusInitialization
from clustering.kmeans import KMeans


df = pd.read_csv("data/iris.csv")

initializer = KmeansPlusPlusInitialization(n_clusters=3)
kmeans = KMeans(initializer=initializer)
clusterer = ClusterDataframe(kmeans=kmeans)
labels = clusterer.cluster(df=df.iloc[:, :-1])

print(labels)
print(clusterer.kmeans.inertia_, clusterer.kmeans.iterations_)
