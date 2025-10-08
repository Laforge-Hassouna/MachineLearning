import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler

def load_arff_data(file_path):
    """Load numeric data from .arff and return as NumPy array."""
    data, meta = arff.loadarff(file_path)
    X = np.array([list(row)[:-1] for row in data], dtype=float)  
    return X

def plot_clusters(data, labels, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=50, alpha=0.8, edgecolor='k')
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    arff_file = os.path.join('dataset', 'artificial', 'diamond9.arff')
    X = load_arff_data(arff_file)
    print("Shape du dataset :", X.shape)

    import opti_k_means
    kmeans, k = opti_k_means.elbow_kmeans(X)
    labels = kmeans.predict(X)

    plot_clusters(X, labels, f"KMeans (k={k})")
