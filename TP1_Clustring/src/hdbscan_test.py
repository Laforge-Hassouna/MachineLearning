import os
import time
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
import hdbscan


# ----------- Fonction pour charger ARFF -----------
def load_arff(path):
    data, meta = arff.loadarff(path)
    X = np.array([list(row)[:-1] for row in data], dtype=float)
    return X


# ----------- Fonction générique d'évaluation -----------
def evaluate_clustering(X, labels):
    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)

    if n_clusters < 2:
        return None  # pas de vrai clustering

    return {
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Silhouette": silhouette_score(X, labels)
    }


# ----------- Fonction DBSCAN + mesure temps -----------
def run_dbscan(X, eps, min_samples):
    start = time.time()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    duration = (time.time() - start) * 1000  # ms

    scores = evaluate_clustering(X, labels)
    if scores is None:
        return None

    scores["Temps (ms)"] = duration
    return scores


# ----------- Fonction HDBSCAN + mesure temps -----------
def run_hdbscan(X, min_cluster_size):
    start = time.time()
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
    duration = (time.time() - start) * 1000  # ms

    scores = evaluate_clustering(X, labels)
    if scores is None:
        return None

    scores["Temps (ms)"] = duration
    return scores


# ----------- TABLEAU FINAL (adaptable) -----------

results = []

datasets = [
    ("xclara.arff",        "DBSCAN", {"eps": 0.6,  "min_samples": 8}),
    ("xclara.arff",        "HDBSCAN", {"min_cluster_size": 6}),

    ("sizes3.arff",        "DBSCAN", {"eps": 1.0,  "min_samples": 20}),
    ("sizes3.arff",        "HDBSCAN", {"min_cluster_size": 10}),

    ("long2.arff",         "DBSCAN", {"eps": 0.5,  "min_samples": 10}),
    ("long2.arff",         "HDBSCAN", {"min_cluster_size": 7}),

    ("sizes4.arff",        "DBSCAN", {"eps": 1.2,  "min_samples": 8}),
    ("sizes4.arff",        "HDBSCAN", {"min_cluster_size": 8}),
]

dataset_folder = "./dataset/artificial/"

for filename, method, params in datasets:
    path = os.path.join(dataset_folder, filename)
    X = load_arff(path)
    X = StandardScaler().fit_transform(X)

    if method == "DBSCAN":
        scores = run_dbscan(X, params["eps"], params["min_samples"])
        label = f"Dbscan {params['eps']} eps, {params['min_samples']} min pts"

    else:  # HDBSCAN
        scores = run_hdbscan(X, params["min_cluster_size"])
        label = f"Hdbscan {params['min_cluster_size']}"

    if scores:
        scores["Méthode"] = label
        scores["Jeu"] = filename.replace(".arff", "")
        results.append(scores)


# ----------- Convertir en tableau -----------

df = pd.DataFrame(results)
df = df[["Méthode", "Davies-Bouldin", "Calinski-Harabasz", "Silhouette", "Temps (ms)", "Jeu"]]

print(df.to_string(index=False))
