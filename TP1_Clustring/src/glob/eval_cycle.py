import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import defaultdict
from Method import Method

def load_arff_data(file_path):
    """Load numeric data from .arff and return as NumPy array."""
    data, meta = arff.loadarff(file_path)
    X = np.array([list(row)[:-1] for row in data], dtype=float)
    return X

def evaluate_clustering(X, labels):
    """Evaluate clustering and return scores."""
    if len(set(labels)) < 2:
        return {"silhouette": -1, "calinski_harabasz": -1, "davies_bouldin": -1}
    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels)
    }

def plot_comparison(scores_dict, dataset_name):
    """Plot comparison of methods for a single dataset."""
    methods = scores_dict.keys()
    methods_names = [str(method) for method in methods]
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(18, 5))
    for i, metric in enumerate(metrics):
        values = [scores_dict[method][metric] for method in methods]
        axes[i].bar(methods_names, values, color=['blue', 'green', 'red', 'purple'])
        axes[i].set_title(f"{dataset_name} - {metric}")
        axes[i].set_ylabel("Score")
        for j, v in enumerate(values):
            axes[i].text(j, v, f"{v:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def plot_global_comparison(all_scores):
    """Plot global comparison of methods across all datasets."""
    methods = list(next(iter(all_scores.values())).keys())
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin"]
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 12))
    for i, metric in enumerate(metrics):
        dataset_names = list(all_scores.keys())
        width = 0.2
        x = np.arange(len(dataset_names))
        for j, method in enumerate(methods):
            values = [all_scores[ds][method][metric] for ds in dataset_names]
            axes[i].bar(x + j*width, values, width=width, label=method)
        axes[i].set_title(metric)
        axes[i].set_xticks(x + width*(n_methods-1)/2)
        axes[i].set_xticklabels(dataset_names, rotation=45)
        axes[i].set_ylabel("Score")
        axes[i].legend()
    plt.tight_layout()
    plt.show()

from kmeans import KMeansMethod
from agglo import AgglomerativeMethod

def main():
    dataset_folder = "dataset/artificial"
    methods: list[Method] = [
        KMeansMethod(),
        AgglomerativeMethod('single'),
        AgglomerativeMethod('average'),
        AgglomerativeMethod('complete'),
        AgglomerativeMethod('ward')
    ]
    all_scores = defaultdict(dict)

    for arff_file in os.listdir(dataset_folder):
        if arff_file.endswith(".arff"):
            file_path = os.path.join(dataset_folder, arff_file)
            X = load_arff_data(file_path)
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            dataset_name = arff_file.split(".")[0]
            print(f"\nProcessing {dataset_name}...")

            scores = {}
            for method in methods:
                labels = method.predict(X)
                scores[method] = evaluate_clustering(X, labels)

            all_scores[dataset_name] = scores
            plot_comparison(scores, dataset_name)

    # Global comparison plot
    plot_global_comparison(all_scores)

if __name__ == "__main__":
    main()
