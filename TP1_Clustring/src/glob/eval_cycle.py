import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
from Method import Method

from kmeans import KMeansMethod
from agglo import AgglomerativeMethod

# -------------------------------------------------------------------
# 1. Chargement des données
# -------------------------------------------------------------------
def load_arff_data(file_path):
    """Load numeric data from .arff and return as NumPy array."""
    data, meta = arff.loadarff(file_path)
    # On enlève la dernière colonne si c'est le label
    X = np.array([list(row)[:-1] for row in data], dtype=float)
    return X

# -------------------------------------------------------------------
# 2. Évaluation des clusters
# -------------------------------------------------------------------
def evaluate_clustering(X, labels):
    """Evaluate clustering and return scores.
    Retourne np.nan quand les métriques ne sont pas définies (moins de 2 clusters)."""
    n_clusters = len(set(labels))

    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan
        }

    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels)
    }

# -------------------------------------------------------------------
# 3. Plots par dataset
# -------------------------------------------------------------------
def plot_comparison(scores_dict, dataset_name):
    """Plot comparison of methods for a single dataset."""
    methods = list(scores_dict.keys())
    methods_names = [str(method) for method in methods]
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]

    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 5))

    # Palette auto en utilisant tab10
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    for i, metric in enumerate(metrics):
        values = [scores_dict[m][metric] for m in methods]

        axes[i].bar(methods_names, values, color=colors)
        axes[i].set_title(f"{dataset_name} - {metric}")
        axes[i].set_ylabel("Score")
        axes[i].set_xticklabels(methods_names, rotation=15, ha='right')

        # Affichage des valeurs au-dessus des barres
        for j, v in enumerate(values):
            if np.isnan(v):
                label = "NA"
                y = 0
            else:
                label = f"{v:.3f}"
                y = v
            axes[i].text(j, y, label, ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 4. Plot global
# -------------------------------------------------------------------
def plot_global_comparison(all_scores):
    """Plot global comparison of methods across all datasets."""
    dataset_names = list(all_scores.keys())
    if not dataset_names:
        print("Aucun score à afficher.")
        return

    methods = list(next(iter(all_scores.values())).keys())
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]

    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 12))
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    x = np.arange(len(dataset_names))
    width = 0.8 / n_methods  # largeur des barres

    for i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            values = [all_scores[ds][method][metric] for ds in dataset_names]
            axes[i].bar(
                x + j * width,
                values,
                width=width,
                label=str(method),
                color=colors[j]
            )

        axes[i].set_title(metric)
        axes[i].set_xticks(x + width * (n_methods - 1) / 2)
        axes[i].set_xticklabels(dataset_names, rotation=45, ha='right')
        axes[i].set_ylabel("Score")
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------
# 5. Main
# -------------------------------------------------------------------
def main():
    dataset_folder = "dataset/artificial"

    # Nombre de clusters à utiliser pour TOUTES les méthodes
    N_CLUSTERS = 4

    methods: list[Method] = [
        KMeansMethod(n_clusters=N_CLUSTERS),
        AgglomerativeMethod(linkage='single',   n_clusters=N_CLUSTERS),
        AgglomerativeMethod(linkage='average',  n_clusters=N_CLUSTERS),
        AgglomerativeMethod(linkage='complete', n_clusters=N_CLUSTERS),
        AgglomerativeMethod(linkage='ward',     n_clusters=N_CLUSTERS),
    ]

    all_scores = defaultdict(dict)

    for arff_file in os.listdir(dataset_folder):
        if not arff_file.endswith(".arff"):
            continue

        file_path = os.path.join(dataset_folder, arff_file)
        X = load_arff_data(file_path)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        dataset_name = arff_file.split(".")[0]
        print(f"\nProcessing {dataset_name}...")

        scores = {}
        for method in methods:
            labels = method.predict(X_scaled)
            scores[method] = evaluate_clustering(X_scaled, labels)

        all_scores[dataset_name] = scores
        plot_comparison(scores, dataset_name)

    # Global comparison plot
    plot_global_comparison(all_scores)


if __name__ == "__main__":
    main()
