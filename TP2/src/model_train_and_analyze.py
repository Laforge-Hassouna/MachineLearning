import pandas as pd
import numpy as np
import sys
import os
import time
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# =====================================================
# Directories
# =====================================================

OUTPUT_DIR = "tests"
EXP5_DIR = "tests/exp5"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXP5_DIR, exist_ok=True)

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
# Exploratory Data Analysis (EDA)
# =====================================================

def plot_correlation_matrix(X, filepath):
    corr = X.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# =====================================================
# Metrics & plots
# =====================================================

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
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

def plot_roc_curve(y_true, y_score, title, filepath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return roc_auc

def plot_pr_curve(y_true, y_score, title, filepath):
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_threshold_metrics(y_true, y_score, title, filepath):
    thresholds = np.linspace(0.05, 0.95, 19)
    accs, precs, recs, f1s = [], [], [], []

    for thr in thresholds:
        y_pred_thr = (y_score >= thr).astype(int)
        accs.append(accuracy_score(y_true, y_pred_thr))
        precs.append(precision_score(y_true, y_pred_thr, zero_division=0))
        recs.append(recall_score(y_true, y_pred_thr, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_thr, zero_division=0))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, accs, label="Accuracy")
    plt.plot(thresholds, precs, label="Precision")
    plt.plot(thresholds, recs, label="Recall")
    plt.plot(thresholds, f1s, label="F1-score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# =====================================================
# Full evaluation pipeline
# =====================================================

def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, output_dir):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred)

    save_confusion_matrix(
        metrics["cm"],
        f"{model_name} – Confusion Matrix (Test)",
        f"{output_dir}/cm_{model_name.lower()}.png"
    )

    roc_auc = plot_roc_curve(
        y_test, y_score,
        f"{model_name} – ROC Curve",
        f"{output_dir}/roc_{model_name.lower()}.png"
    )

    plot_pr_curve(
        y_test, y_score,
        f"{model_name} – Precision-Recall Curve",
        f"{output_dir}/pr_{model_name.lower()}.png"
    )

    plot_threshold_metrics(
        y_test, y_score,
        f"{model_name} – Threshold Analysis",
        f"{output_dir}/threshold_{model_name.lower()}.png"
    )

    return metrics, roc_auc, train_time

# =====================================================
# Grid Search
# =====================================================

def run_grid_search(model, param_grid, X_train, y_train):
    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

# =====================================================
# Main
# =====================================================

def main():
    if len(sys.argv) < 4:
        print("Usage: python model_train_and_analyze.py <features.csv> <labels.csv> <test_size>")
        sys.exit(1)

    features_path = sys.argv[1]
    labels_path = sys.argv[2]
    test_size = float(sys.argv[3])

    print("📥 Loading data...")
    X, y = read_features_and_labels(features_path, labels_path)

    print("📊 Running EDA...")
    plot_correlation_matrix(X, f"{EXP5_DIR}/correlation_matrix.png")

    print("✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    models = {
        "RandomForest": RandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    for name, model in models.items():
        print(f"\n🚀 Evaluating {name}")
        metrics, roc_auc, train_time = evaluate_model(
            model, name, X_train, y_train, X_test, y_test, EXP5_DIR
        )

        print(f"{name} | Acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f} | AUC={roc_auc:.4f} | Time={train_time:.2f}s")

    print("\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()
