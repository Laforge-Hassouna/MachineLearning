import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Import de ton préprocessing
from preproc1 import preprocess_data

# =========================
# 1. Charger les données
# =========================
X, y = preprocess_data(
    "Dataset/alt_acsincome_ca_features_85.csv",
    "Dataset/alt_acsincome_ca_labels_85.csv"
)

# =========================
# 2. Split train/test (75/25)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, shuffle=True
)

print("---- Taille du jeu de données ----")
print("Train :", X_train.shape[0])
print("Test  :", X_test.shape[0])

# =========================
# 3. Fonction d’évaluation
# =========================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    cpu_time = time.perf_counter() - t0

    # train
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    train_conf = confusion_matrix(y_train, train_pred)

    # test
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_conf = confusion_matrix(y_test, test_pred)

    return train_acc, test_acc, train_conf, test_conf, cpu_time

# =========================
# 4. Modèles
# =========================
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

# =========================
# 5. Évaluation
# =========================
print("\n===== Résultats Expe 1 =====")

for name, model in models.items():
    print(f"\n---- {name} ----")
    train_acc, test_acc, train_cm, test_cm, cpu_time = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )

    print("Train accuracy :", train_acc)
    print("Test accuracy  :", test_acc)
    print("Temps CPU (s)  :", cpu_time)
    print("Matrice confusion train :\n", train_cm)
    print("Matrice confusion test :\n", test_cm)
