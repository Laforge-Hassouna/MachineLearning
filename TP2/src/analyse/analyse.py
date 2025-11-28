import pandas as pd
import numpy as np
import sys
import os
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------

def read_features_and_labels(features_path, labels_path):
    """
    Read features and labels from separate CSV files.
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    # Flatten labels column if needed
    if y.shape[1] == 1:
        y = y.iloc[:, 0]

    return X, y


# --------------------------------------------------------
# 2. Training and evaluation
# --------------------------------------------------------

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Train a model, compute training time, accuracy and confusion matrix.
    """
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return acc, cm, end - start


def run_default_models(X_train, y_train, X_test, y_test):

    models = {
        "RandomForest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n➡️ Training default {name}...")
        acc, cm, t = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        results[name] = (acc, cm, t)

        print(f"{name} accuracy: {acc:.4f}")
        print(f"{name} training time: {t:.2f}s")
        print(f"{name} confusion matrix:\n{cm}")

    return results


# --------------------------------------------------------
# 3. Grid search
# --------------------------------------------------------

def run_grid_search(model, param_grid, X_train, y_train, X_test, y_test, model_name):

    print(f"\n🔍 Running GridSearchCV for {model_name}...\n")

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    start = time.time()
    grid.fit(X_train, y_train)
    end = time.time()

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Best params: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Training + tuning time: {end - start:.2f}s")
    print(f"Confusion matrix:\n{cm}")

    return best, acc, cm


def run_grid_search_all(X_train, y_train, X_test, y_test):

    grids = {
        "RandomForest": (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20]
            }
        ),

        "AdaBoost": (
            AdaBoostClassifier(),
            {
                "n_estimators": [50, 200],
                "learning_rate": [0.5, 1.0, 2.0]
            }
        ),

        "XGBoost": (
            XGBClassifier(eval_metric="logloss"),
            {
                "n_estimators": [100, 300],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.05, 0.1, 0.2]
            }
        )
    }

    results = {}

    for name, (model, grid) in grids.items():
        best, acc, cm = run_grid_search(model, grid, X_train, y_train, X_test, y_test, name)
        results[name] = (best, acc, cm)

    return results


# --------------------------------------------------------
# 4. CLI and module usage
# --------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python model_train_and_tune.py <features.csv> <labels.csv> <test_size>")
        sys.exit(1)

    features_path = sys.argv[1]
    labels_path = sys.argv[2]
    test_size = float(sys.argv[3])

    print("📥 Loading data...")
    X, y = read_features_and_labels(features_path, labels_path)

    print("✂️ Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    print("\n=== Training default models ===")
    run_default_models(X_train, y_train, X_test, y_test)

    print("\n=== Running hyperparameter tuning ===")
    run_grid_search_all(X_train, y_train, X_test, y_test)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
