import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from Method import Method


class AgglomerativeMethod(Method):

    def __init__(self, linkage):
        self.linkage = linkage

    def predict(self, X):
        distances = pairwise_distances(X)
        # On prend un seuil basé sur la distance moyenne + 1 écart-type
        distance_threshold = np.mean(distances) + np.std(distances)

        model = AgglomerativeClustering(
            linkage=self.linkage,
            distance_threshold=distance_threshold,
            n_clusters=None
        )
        labels = model.fit_predict(X)
        return labels

    def __repr__(self):
        return f"agglo-clustering ({self.linkage})"
