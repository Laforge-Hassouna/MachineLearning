import pandas as pd
import sys
import time
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


# =====================================================
# Directories
# =====================================================

OUTPUT_DIR = "tests"
EXP2_DIR = "tests/exp2"
EXP3_DIR = "tests/exp3"
EXP4_DIR = "tests/exp4"
EXP5_DIR = "tests/exp5"

for d in [OUTPUT_DIR, EXP2_DIR, EXP3_DIR, EXP4_DIR, EXP5_DIR]:
    os.makedirs(d, exist_ok=True)

os.makedirs(f"{EXP4_DIR}/CO", exist_ok=True)
os.makedirs(f"{EXP4_DIR}/NE", exist_ok=True)


# =====================================================
# Data loading
# =====================================================

def read_features_and_labels(features_path, labels_path):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    return X, y


# =====================================================
# Metrics
# =====================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred)
    }


def save_confusion_matrix(cm, title, filepath):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# =====================================================
# Expérimentation 1 – modèles par défaut
# =====================================================

def train_and_evaluate_default(model, model_name, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    save_confusion_matrix(
        metrics["cm"],
        f"{model_name} – Confusion Matrix (Test)",
        f"{OUTPUT_DIR}/confusion_matrix_{model_name.lower()}.png"
    )

    return metrics, train_time


# =====================================================
# Expérimentation 2 – GridSearch
# =====================================================

def run_grid_search_and_evaluate(model, param_grid, model_name,
                                 X_train, y_train, X_test, y_test):

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=2,
        verbose=1
    )

    start = time.time()
    grid.fit(X_train, y_train)
    total_time = time.time() - start

    best = grid.best_estimator_

    train_pred = best.predict(X_train)
    test_pred = best.predict(X_test)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    save_confusion_matrix(
        test_metrics["cm"],
        f"{model_name} – Confusion Matrix (Test)",
        f"{EXP2_DIR}/confusion_matrix_{model_name.lower()}.png"
    )

    save_confusion_matrix(
        train_metrics["cm"],
        f"{model_name} – Confusion Matrix (Train)",
        f"{EXP3_DIR}/confusion_matrix_train_{model_name.lower()}.png"
    )

    return {
        "best_model": best,
        "best_params": grid.best_params_,
        "train": train_metrics,
        "test": test_metrics,
        "time": total_time
    }


# =====================================================
# Expérimentation 5 – impact taille dataset
# =====================================================

def evaluate_on_fixed_test(model, X_train_sub, y_train_sub, X_test, y_test):
    start = time.time()
    model.fit(X_train_sub, y_train_sub)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    metrics["time"] = train_time
    return metrics


def run_experiment_5(best_models, X_train, y_train, X_test, y_test,
                     train_fracs, repeats=3):

    results = {name: [] for name in best_models}
    n_total = len(X_train)

    for frac in train_fracs:
        n_sub = int(n_total * frac)

        # ============================
        # Cas spécial : 100 % des données
        # ============================
        if frac == 1.0:
            for name, base_model in best_models.items():
                model = base_model.__class__(**base_model.get_params())
                m = evaluate_on_fixed_test(model, X_train, y_train, X_test, y_test)

                results[name].append({
                    "n_train": n_total,
                    "accuracy_mean": m["accuracy"],
                    "accuracy_std": 0.0,
                    "precision_mean": m["precision"],
                    "precision_std": 0.0,
                    "recall_mean": m["recall"],
                    "recall_std": 0.0,
                    "time_mean": m["time"],
                    "time_std": 0.0
                })
            continue

        # ============================
        # Cas général : sous-échantillonnage
        # ============================
        if n_sub < 50:
            continue

        splitter = StratifiedShuffleSplit(
            n_splits=repeats,
            train_size=n_sub,
            random_state=42
        )

        for name, base_model in best_models.items():
            accs, precs, recs, times = [], [], [], []

            for train_idx, _ in splitter.split(X_train, y_train):
                X_sub = X_train.iloc[train_idx]
                y_sub = y_train.iloc[train_idx]

                model = base_model.__class__(**base_model.get_params())
                m = evaluate_on_fixed_test(model, X_sub, y_sub, X_test, y_test)

                accs.append(m["accuracy"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                times.append(m["time"])

            results[name].append({
                "n_train": n_sub,
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "precision_mean": np.mean(precs),
                "precision_std": np.std(precs),
                "recall_mean": np.mean(recs),
                "recall_std": np.std(recs),
                "time_mean": np.mean(times),
                "time_std": np.std(times)
            })

    return results


def plot_exp5_curves(exp5_results, metric, outpath):
    plt.figure(figsize=(7, 5))
    for model, rows in exp5_results.items():
        xs = [r["n_train"] for r in rows]
        ys = [r[f"{metric}_mean"] for r in rows]
        yerr = [r[f"{metric}_std"] for r in rows]
        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=model)

    plt.xlabel("Taille entraînement")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    if len(sys.argv) != 4:
        print("Usage: python model_train_and_tune.py <features.csv> <labels.csv> <test_size>")
        sys.exit(1)

    features_path = sys.argv[1]
    labels_path = sys.argv[2]
    test_size = float(sys.argv[3])

    print("📥 Loading data...")
    X, y = read_features_and_labels(features_path, labels_path)

    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # =================================================
    # EXPÉRIMENTATION 1 – Modèles par défaut
    # =================================================
    print("\n===============================")
    print(" EXPÉRIMENTATION 1 : Modèles par défaut ")
    print("===============================\n")

    default_models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
    }

    for name, model in default_models.items():
        metrics_test, train_time = train_and_evaluate_default(
            model, name, X_train, y_train, X_test, y_test
        )

        print(
            f"{name} | "
            f"Accuracy={metrics_test['accuracy']:.4f} | "
            f"Precision={metrics_test['precision']:.4f} | "
            f"Recall={metrics_test['recall']:.4f} | "
            f"Time={train_time:.2f}s"
        )

    # =================================================
    # EXPÉRIMENTATION 2 – GridSearchCV
    # =================================================
    print("\n===============================")
    print(" EXPÉRIMENTATION 2 : GridSearchCV ")
    print("===============================\n")

    param_grids = {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 400],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 3, 5, 10]
            }
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {"n_estimators": [50, 200], "learning_rate": [0.5, 1.0, 2.0]}
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42),
            {"n_estimators": [100, 300], "max_depth": [3, 6], "learning_rate": [0.05, 0.1]}
        )
    }

    for name, (model, grid) in param_grids.items():

        # Infos protocole
        n_combinations = np.prod([len(v) for v in grid.values()])
        n_folds = 3
        n_trainings = n_combinations * n_folds

        print(f"--- {name} ---")
        print("Validation croisée")
        print(f"— Nombre de plis : {n_folds}")
        print(f"— Nombre de combinaisons testées : {n_combinations}")
        print(f"— Nombre total d’entraînements : {n_trainings}")

        res = run_grid_search_and_evaluate(
            model, grid, name, X_train, y_train, X_test, y_test
        )

        print("Performances")
        print(f"— Accuracy entraînement : {res['train']['accuracy']:.4f}")
        print(f"— Accuracy test        : {res['test']['accuracy']:.4f}")
        print(f"— Temps de calcul total : {res['time']:.2f} s\n")


if __name__ == "__main__":
    main()
