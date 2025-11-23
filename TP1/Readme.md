| Méthode                  | Davies-Bouldin | Calinski-Harabasz | Silhouette | Temps (ms)    | Jeu de Données |
|--------------------------|----------------|--------------------|------------|----------------|-----------------|
| **Dbscan 6 eps, 8 min pts** | 1.76           | 1750               | 0.44       | 1750 ms        | xclara          |
| **Hdbscan 6**            | 1.57           | 6412               | 0.65       | 3702.18 ms     | xclara          |
|                          |                |                    |            |                |                 |
| **Dbscan 1 eps, 20 min** | 1.25           | 514.57             | 0.42       | 599.71 ms      | sizes3          |
| **Hdbscan 10**           | 1.56           | 461.86             | 0.48       | 447.11 ms      | sizes3          |
|                          |                |                    |            |                |                 |
| **Dbscan**               | 0.31           | 125                | 0.51       | 2033.09 ms     | long2           |
| **Hdbscan 7**            | 1.99           | 119                | 0.29       | 2562.36 ms     | long2           |
|                          |                |                    |            |                |                 |
| **Dbscan 1.2 eps pour 8**| 1.49           | 381.46             | 0.47       | 1006.2 ms      | sizes4          |
| **Hdbscan 8**            | 1.43           | 433.62             | 0.49       | 2602.85 ms     | sizes4          |
