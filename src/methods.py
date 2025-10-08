import os
import arff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_arff_data(file_path):
    """Load data from a .arff file and return as a numpy array."""
    with open(file_path, 'r') as f:
        data = arff.load(f)
    return np.array(data['data'])

def plot_clusters(data, labels, k):
    """Plot the clustered data, coloring each point by its cluster."""
    plt.figure(figsize=(8, 6))
    # Use a colormap with enough distinct colors for all clusters
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7)
    plt.title(f'K-Means Clustering (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster ID')
    plt.grid(True)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your .arff file path
    arff_file = os.path.join('dataset', 'artificial', '2d-10c.arff')
    data = load_arff_data(arff_file)

    # Number of clusters
    k = 3

    # Apply K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)

    # Plot the result
    plot_clusters(data, labels, k)