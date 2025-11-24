import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
import sys
import pandas as pd
import hdbscan
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from itertools import product
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from analysis_plots import plot_hdbscan_condensed_tree, plot_silhouette_clusters

# ==========================================================
# === PARAMÈTRES GÉNÉRAUX
# ==========================================================
DATASETS_DIR = "dataset/artificial"
RESULTS_PATH = "../resultats/HDBSCAN"
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")
FULL_SCORES_PATH = os.path.join(RESULTS_PATH, "full_scores")
GLOBAL_CSV_PATH = os.path.join(RESULTS_PATH, "best_scores.csv")

# Création des dossiers nécessaires
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(FULL_SCORES_PATH, exist_ok=True)

# Liste des fichiers à ignorer
IGNORE_DATASETS = ["birch-rg1.arff", "birch-rg3.arff"]

# Initialisation du CSV global s’il n’existe pas
if not os.path.exists(GLOBAL_CSV_PATH):
    pd.DataFrame(columns=[
        "dataset",
        "metric_eval",
        "min_cluster_size",
        "min_samples",
        "metric",
        "cluster_selection_method",
        "score"
    ]).to_csv(GLOBAL_CSV_PATH, index=False)


# ==========================================================
# === 1. CHARGEMENT DU DATASET
# ==========================================================
def load_dataset(file_path):
    """Charge un fichier ARFF et retourne un tableau numpy."""
    data, _ = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    # On garde uniquement les colonnes numériques
    df = df.select_dtypes(exclude=["object"])
    return df.values


# ==========================================================
# === 2. TEST DE TOUTES LES COMBINAISONS DE PARAMÈTRES
# ==========================================================
def test_hyperparameters(X):
    """Teste toutes les combinaisons de paramètres pour HDBSCAN."""
    min_cluster_size_list = [2, 5, 10]
    min_samples_list = [None, 5, 10]
    metric_list = ["euclidean", "manhattan"]
    cluster_selection_method_list = ["eom", "leaf"]

    results = []

    for min_cluster_size, min_samples, metric, method in product(
        min_cluster_size_list, min_samples_list, metric_list, cluster_selection_method_list
    ):
        try:
            clustering = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                cluster_selection_method=method
            )
            labels = clustering.fit_predict(X)
        except Exception as e:
            print(
                f" Skipping: size={min_cluster_size}, samples={min_samples}, "
                f"metric={metric}, method={method} ({e})"
            )
            continue

        n_unique = len(np.unique(labels))
        if n_unique <= 1 or n_unique == len(X):
            print(f" Ignored: produit {n_unique} clusters")
            continue

        try:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
        except Exception:
            print(
                f" Scores non calculables pour size={min_cluster_size}, metric={metric}"
            )
            continue

        results.append({
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_selection_method": method,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db
        })

    return pd.DataFrame(results)


# ==========================================================
# === 3. ENREGISTREMENT DES RÉSULTATS
# ==========================================================
def save_full_results(dataset_name, results_df):
    """Sauvegarde le CSV complet des scores."""
    csv_path = os.path.join(FULL_SCORES_PATH, f"{dataset_name}_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f" Résultats complets sauvegardés : {csv_path}")


# ==========================================================
# === 4. EXTRACTION DES MEILLEURS MODÈLES
# ==========================================================
def get_best_models(results_df):
    """Retourne les meilleures configurations pour chaque métrique."""
    return {
        "silhouette": results_df.loc[results_df["silhouette"].idxmax()],
        "calinski_harabasz": results_df.loc[results_df["calinski_harabasz"].idxmax()],
        "davies_bouldin": results_df.loc[results_df["davies_bouldin"].idxmin()],
    }


# ==========================================================
# === 5. PLOTS DES MEILLEURS MODÈLES
# ==========================================================
def plot_clusters(X, dataset_name, params, metric_name):
    """Génère et enregistre un scatter plot des clusters."""
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=None if pd.isna(params["min_samples"]) else int(params["min_samples"]),
        metric=params["metric"],
        cluster_selection_method=params["cluster_selection_method"]
    ).fit(X)

    labels = clustering.labels_

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=50)
    plt.title(
        f"{dataset_name} — {metric_name} "
        f"(size={params['min_cluster_size']}, samples={params['min_samples']}, "
        f"{params['metric']}, {params['cluster_selection_method']})"
    )
    plt.tight_layout()

    metric_plot_dir = os.path.join(PLOTS_PATH, metric_name)
    os.makedirs(metric_plot_dir, exist_ok=True)

    filename = f"{dataset_name}_{metric_name}.png"
    plt.savefig(os.path.join(metric_plot_dir, filename), dpi=300)
    plt.close()

    print(f" Plot '{metric_name}' sauvegardé : {filename}")


def plot_hdbscan_analysis(X, dataset_name, params, metric_name, results_df=None):
    """
    Génère tous les plots d'analyse pour HDBSCAN :
    - Silhouette plot
    - Condensed tree
    """
    clustering = hdbscan.HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=None if pd.isna(params["min_samples"]) else int(params["min_samples"]),
        metric=params["metric"],
        cluster_selection_method=params["cluster_selection_method"]
    ).fit(X)
    labels = clustering.labels_

    metric_dir = os.path.join(PLOTS_PATH, metric_name)
    analysis_dir = os.path.join(metric_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # 2️ Silhouette
    sil_dir = os.path.join(analysis_dir, "silhouette")
    os.makedirs(sil_dir, exist_ok=True)
    plot_silhouette_clusters(
        X, labels,
        savepath=os.path.join(sil_dir, f"{dataset_name}_silhouette.png")
    )

    # 3️ Condensed tree
    tree_dir = os.path.join(analysis_dir, "condensed_tree")
    os.makedirs(tree_dir, exist_ok=True)
    plot_hdbscan_condensed_tree(
        clustering,
        savepath=os.path.join(tree_dir, f"{dataset_name}_condensed_tree.png")
    )


# ==========================================================
# === 6. MISE À JOUR DU CSV GLOBAL
# ==========================================================
def update_global_csv(dataset_name, best_models):
    """Ajoute les meilleurs modèles dans le CSV global."""
    global_df = pd.read_csv(GLOBAL_CSV_PATH)
    new_rows = []

    for metric_name, params in best_models.items():
        new_rows.append({
            "dataset": dataset_name,
            "metric_eval": metric_name,
            "min_cluster_size": int(params["min_cluster_size"]),
            "min_samples": params["min_samples"],
            "metric": params["metric"],
            "cluster_selection_method": params["cluster_selection_method"],
            "score": params[metric_name]
        })

    global_df = pd.concat([global_df, pd.DataFrame(new_rows)], ignore_index=True)
    global_df.to_csv(GLOBAL_CSV_PATH, index=False)
    print(f" Meilleurs scores ajoutés à {GLOBAL_CSV_PATH}")


# ==========================================================
# === 7. PIPELINE PRINCIPAL
# ==========================================================
def run_hdbscan_experiment(file_path):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n Traitement du dataset : {dataset_name}")

    X = load_dataset(file_path)
    results_df = test_hyperparameters(X)
    save_full_results(dataset_name, results_df)
    best_models = get_best_models(results_df)

    for metric_name, params in best_models.items():
        plot_clusters(X, dataset_name, params, metric_name)
        plot_hdbscan_analysis(X, dataset_name, params, metric_name, results_df)

    update_global_csv(dataset_name, best_models)


# ==========================================================
# === 8. EXÉCUTION UNIQUEMENT SUR cluto-t5-8k.arff
# ==========================================================
if __name__ == "__main__":
    dataset_name = "cluto-t5-8k.arff"
    file_path = os.path.join(DATASETS_DIR, dataset_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    run_hdbscan_experiment(file_path)
    print("\nTraitement HDBSCAN terminé pour cluto-t5-8k.arff.")

