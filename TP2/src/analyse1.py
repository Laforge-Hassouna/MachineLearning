# analyse.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Charger le prétraitement
from preproc1 import preprocess_data

# 2. Charger les données
X, y = preprocess_data("../Dataset/alt_acsincome_ca_features_85.csv", "../Dataset/alt_acsincome_ca_labels_85.csv")
print("X après preprocess :", X.shape)
print("y après preprocess :", y.shape)
# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

# 4. Définir les modèles
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
}

# 5. Entraîner, prédire, évaluer
results = {}

for name, model in models.items():
    print(f"\n====== {name} ======")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)

    results[name] = {
        "accuracy": acc,
        "confusion_matrix": cm,
        "report": report
    }

    # Matrice de confusion plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de confusion – {name}")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

# 6. Comparer les accuracy
plt.figure(figsize=(6, 4))
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]

sns.barplot(x=model_names, y=accuracies)
plt.ylim(0.5, 1.0)
plt.ylabel("Accuracy")
plt.title("Comparaison des modèles – Accuracy")
plt.tight_layout()
plt.savefig("comparaison_accuracy.png")
plt.show()
