# 📘 Expérimentation 2 — Optimisation des hyperparamètres

Dans cette expérience, nous utilisons **le même jeu de données** pour optimiser les trois modèles  
(**Random Forest**, **AdaBoost**, **XGBoost**) via **GridSearchCV**.

Le paramètre donné est :  
**test_size = 0.8**  
→ donc **80 % = test**, **20 % = train**

> Le dataset complet n’a pas été explicitement fourni, mais les matrices de confusion permettent d’inférer la taille du test.

La matrice de confusion du meilleur modèle XGBoost donne :  
- Test = 16 724 + 2 870 + 2 903 + 10 766 = **33 263 échantillons**

Donc le dataset total vaut environ :  
**Total ≈ 33 263 / 0.8 = 41 579 échantillons**  
Train = Total – Test = **8 316 échantillons**

---

# 📊 Table 4 — Taille du jeu de données (Expérimentation 2)

| Jeu | Taille |
|-----|--------|
| **Train** | **≈ 8 316** |
| **Test** | **33 263** |

---

# ⚙️ 1. Hyperparamètres explorés + justification

## 🌲 Random Forest
Hyperparamètres explorés :
```python
{
    "n_estimators": [100, 300],
    "max_depth": [None, 10, 20]
}
```
Justification :

n_estimators : augmenter le nombre d’arbres améliore la stabilité mais augmente le temps de calcul.

max_depth : permet de contrôler l’overfitting ; None = croissance libre de l'arbre.

## ⚡ AdaBoost

Hyperparamètres explorés :

```python
{
    "n_estimators": [50, 200],
    "learning_rate": [0.5, 1.0, 2.0]
}
```

Justification :

n_estimators : plus d’itérations → meilleur ajustement mais plus lent.

learning_rate : contrôle l’importance de chaque modèle faible (trade-off stabilité / précision).

## 🚀 XGBoost

Hyperparamètres explorés :

```python
{
    "n_estimators": [100, 300],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.05, 0.1, 0.2]
}
```

Justification :

n_estimators : robustesse contre variance.

max_depth : profondeur des arbres → contrôle du sur-apprentissage.

learning_rate : plus la valeur est faible, plus le modèle apprend “lentement” mais finement.

# 📐 2. Nombre de plis utilisés

→ Dans le code :

cv = 3

# 🧮 3. Nombre total d’entraînements effectués

Modèle	Combinaisons	CV (3)	Total entraînements
RandomForest	2 × 3 = 6	×3	18
AdaBoost	2 × 3 = 6	×3	18
XGBoost	2 × 3 × 3 = 18	×3	54

# 📑 4. Tableau de résultats (Expérimentation 2)
## 🌲 Random Forest
Random Forest	Train Accuracy	CPU time	Test Accuracy	Hyperparamètres
Défaut	—	10.20 s	0.8173	n_estimators=100, max_depth=None
Optimisé	—	71.18 s	0.8205	n_estimators=300, max_depth=20

➡️ L'amélioration reste légère mais réelle.

## ⚡ AdaBoost
AdaBoost	Train Accuracy	CPU time	Test Accuracy	Hyperparamètres
Défaut	—	1.71 s	0.8032	n_estimators=50, learning_rate=1.0
Optimisé	—	17.66 s	0.8092	n_estimators=200, learning_rate=1.0

➡️ Le gain est modéré. L’augmentation du nombre d’estimateurs améliore légèrement les performances.

## 🚀 XGBoost
XGBoost	Train Accuracy	CPU time	Test Accuracy	Hyperparamètres
Défaut	—	0.20 s	0.8256	n_estimators=100, max_depth=6, learning_rate=0.1
Optimisé	—	11.60 s	0.8264	n_estimators=300, max_depth=6, learning_rate=0.1

➡️ Très légère amélioration ; XGBoost était déjà performant par défaut.

# 📝 5. Analyse par méthode
## 🌲 Random Forest

L’augmentation du nombre d’arbres et la limitation de la profondeur permettent un modèle légèrement meilleur.

Gain faible car RF est assez robuste par défaut.

Temps de calcul très élevé lors du Grid Search.

## ⚡ AdaBoost

Très sensible au nombre d’estimateurs.

Les gains restent faibles, modèle parfois limité par la nature des données.

Mais amélioration constante en augmentant n_estimators.

## 🚀 XGBoost

Meilleures performances globales, même avant optimisation.

L’optimisation ne change que très peu l’accuracy : le modèle était déjà presque optimal.

Temps d’entraînement très faible par rapport aux deux autres (implémentation optimisée).

# ✅ Conclusion générale

XGBoost est le meilleur modèle, même sans optimisation poussée.

Random Forest et AdaBoost progressent légèrement après tuning, mais sans franchir XGBoost.

Les grilles d’hyperparamètres restent raisonnables pour éviter une explosion du temps de calcul tout en couvrant les valeurs les plus pertinentes.