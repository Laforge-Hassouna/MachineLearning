from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluer_clustering(X, labels):
    """
    Évalue une solution de clustering sur un dataset X avec les labels fournis.

    Paramètres:
    - X : tableau numpy ou matrice de données (n_echantillons, n_caracteristiques)
    - labels : tableau numpy des labels de clustering (n_echantillons,)

    Retourne:
    - Un dictionnaire contenant les scores des trois indices.
    """
    # Coefficient de silhouette
    silhouette = silhouette_score(X, labels)

    # Indice de Calinski-Harabasz
    calinski_harabasz = calinski_harabasz_score(X, labels)

    # Indice de Davies-Bouldin
    davies_bouldin = davies_bouldin_score(X, labels)

    return silhouette, calinski_harabasz, davies_bouldin
