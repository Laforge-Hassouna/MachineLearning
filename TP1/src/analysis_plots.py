import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm


def plot_silhouette_clusters(X, labels, savepath=None):
    """
    Trace un vrai diagramme de silhouette pour un clustering donné.
    Les points avec label = -1 (bruit) sont ignorés.
    """
    # On ignore le bruit pour le calcul de la silhouette
    mask = labels != -1
    X_clust = X[mask]
    labels_clust = labels[mask]

    if len(np.unique(labels_clust)) < 2:
        print("Silhouette impossible : moins de 2 clusters (hors bruit).")
        return

    sil_values = silhouette_samples(X_clust, labels_clust)
    cluster_labels = np.unique(labels_clust)
    n_clusters = len(cluster_labels)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    y_lower = 10
    for i, c in enumerate(cluster_labels):
        ith_cluster_sil = sil_values[labels_clust == c]
        ith_cluster_sil.sort()
        size_cluster_i = ith_cluster_sil.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.get_cmap("tab10")(i % 10)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(c))
        y_lower = y_upper + 10

    ax1.set_title("Silhouette plot")
    ax1.set_xlabel("Valeur du coefficient de silhouette")
    ax1.set_ylabel("Cluster")

    ax1.axvline(x=np.mean(sil_values), color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_hdbscan_condensed_tree(clusterer, savepath=None):
    """
    Trace le condensed tree de HDBSCAN avec une palette de couleurs valide.
    """
    plt.figure(figsize=(10, 6))

    # Palette de 10 couleurs issues de 'tab10'
    palette = cm.get_cmap("tab10")
    selection_colors = [palette(i) for i in range(10)]

    clusterer.condensed_tree_.plot(
        select_clusters=True,
        selection_palette=selection_colors
    )

    plt.title("HDBSCAN Condensed Tree")

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()
