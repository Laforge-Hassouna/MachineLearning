import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def elbow_kmeans(data, k_max=10):
    """
    Trouve le meilleur k pour k-means en utilisant la méthode du coude.

    Paramètres:
    - data: numpy array ou pandas DataFrame, les données à clusteriser.
    - k_max: int, le nombre maximum de clusters à tester.

    Retourne:
    - kmeans: KMeans, instance d'object KMeans entraînée
    - meilleur_k: int, le k suggéré par la méthode du coude.
    - inerties: list, les inerties pour chaque k (pour tracer le graphique).
    """
    inerties = []
    kmeans_liste: list[KMeans] = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        kmeans_liste.append(kmeans)
        inerties.append(kmeans.inertia_)

    # Calcul des différences pour trouver le "coude"
    diffs = np.diff(inerties)
    diff_ratios = diffs[:-1] / diffs[1:]
    meilleur_k = np.argmax(diff_ratios) + 2  # +2 car diff commence à k=2
    kmeans: KMeans = kmeans_liste[meilleur_k-1]

    return kmeans, meilleur_k