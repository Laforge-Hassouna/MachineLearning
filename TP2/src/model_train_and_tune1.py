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
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
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
os.makedirs(EXP5_DIR, exist_ok=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXP2_DIR, exist_ok=True)
os.makedirs(EXP3_DIR, exist_ok=True)

os.makedirs(EXP4_DIR, exist_ok=True)
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
# Metrics + plots utilities
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


def print_confusion_matrix(cm, model_name, dataset_name, split_name):
    print(f"\nConfusion matrix ({split_name}) – {model_name} on {dataset_name}")
    print(cm)


def plot_roc_curve(y_true, y_score, title, filepath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
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
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
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

    y_pred_test = model.predict(X_test)
    metrics_test = compute_metrics(y_test, y_pred_test)

    save_confusion_matrix(
        metrics_test["cm"],
        f"{model_name} – Confusion Matrix (Test)",
        f"{OUTPUT_DIR}/confusion_matrix_{model_name.lower()}.png"
    )

    return metrics_test, train_time


# =====================================================
# Expérimentation 2 – GridSearch
# =====================================================

def run_grid_search_and_evaluate(
    model,
    param_grid,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    cv=3
):
    print(f"\n GridSearchCV – {model_name}")

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=2,   # plus stable sous WSL; tu peux mettre -1 si tu veux
        verbose=1
    )

    start = time.time()
    grid.fit(X_train, y_train)
    total_time = time.time() - start

    best_model = grid.best_estimator_

    # TRAIN
    y_train_pred = best_model.predict(X_train)
    train_metrics = compute_metrics(y_train, y_train_pred)

    # TEST
    y_test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)

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
        "best_model": best_model,
        "best_params": grid.best_params_,
        "cv": cv,
        "n_trainings": len(grid.cv_results_["params"]) * cv,
        "train": train_metrics,
        "test": test_metrics,
        "time": total_time
    }


# =====================================================
# Expérimentation 3 – plots
# =====================================================

def plot_metric_comparison(exp3_results, metric_name):
    models = list(exp3_results.keys())
    values = [exp3_results[m]["test"][metric_name] for m in models]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=values)
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Expérimentation 3 – {metric_name.capitalize()} (Test)")
    plt.tight_layout()
    plt.savefig(f"{EXP3_DIR}/{metric_name}_comparison_exp3.png")
    plt.close()


# =====================================================
# Expérimentation 4 – inference + curves
# =====================================================

def inference_on_new_dataset(model, model_name, dataset_name, X_new, y_new, output_dir):
    # scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_new)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_new)
    else:
        # fallback
        y_score = model.predict(X_new)

    # default threshold
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_new, y_pred),
        "precision": precision_score(y_new, y_pred, zero_division=0),
        "recall": recall_score(y_new, y_pred, zero_division=0),
        "f1": f1_score(y_new, y_pred, zero_division=0),
        "cm": confusion_matrix(y_new, y_pred)
    }

    # terminal CM
    print_confusion_matrix(metrics["cm"], model_name, dataset_name, "Inference")

    # save CM
    save_confusion_matrix(
        metrics["cm"],
        f"{model_name} – Confusion Matrix (Inference) – {dataset_name}",
        f"{output_dir}/confusion_matrix_{model_name.lower()}.png"
    )

    # ROC + AUC
    metrics["auc"] = plot_roc_curve(
        y_new, y_score,
        title=f"ROC – {model_name} – {dataset_name}",
        filepath=f"{output_dir}/roc_{model_name.lower()}.png"
    )

    # PR curve
    plot_pr_curve(
        y_new, y_score,
        title=f"Precision–Recall – {model_name} – {dataset_name}",
        filepath=f"{output_dir}/pr_{model_name.lower()}.png"
    )

    # threshold metrics
    plot_threshold_metrics(
        y_new, y_score,
        title=f"Threshold vs Metrics – {model_name} – {dataset_name}",
        filepath=f"{output_dir}/threshold_metrics_{model_name.lower()}.png"
    )

    return metrics
def evaluate_on_fixed_test(model, X_train_sub, y_train_sub, X_test, y_test):
    start = time.time()
    model.fit(X_train_sub, y_train_sub)
    train_time = time.time() - start

    # scores proba si dispo (utile AUC)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics["time"] = train_time
    return metrics


def run_experiment_5(best_models, X_train, y_train, X_test, y_test,
                     train_fracs=None, repeats=3, random_state=42):
    """
    best_models: dict {model_name: fitted_or_unfitted_model_with_best_params}
                -> IMPORTANT: on fournira des modèles "neufs" (clone-like), pas déjà fit.
    """

    if train_fracs is None:
        train_fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = {name: [] for name in best_models.keys()}

    sss = StratifiedShuffleSplit(
        n_splits=repeats,
        train_size=None,   # on fixera via indice
        test_size=None,
        random_state=random_state
    )

    # pour stratifier correctement, on va générer des splits à chaque fraction
    for frac in train_fracs:
        n_sub = int(len(X_train) * frac)
        if n_sub < 50:
            continue

        # Générer repeats sous-échantillons stratifiés
        splitter = StratifiedShuffleSplit(
            n_splits=repeats,
            train_size=n_sub,
            random_state=random_state
        )

        for model_name, base_model in best_models.items():
            accs, precs, recs, f1s, times = [], [], [], [], []

            for train_idx, _ in splitter.split(X_train, y_train):
                X_sub = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
                y_sub = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]

                # IMPORTANT: recréer un modèle neuf pour chaque fit
                # (sinon certains modèles gardent l'état)
                model = base_model.__class__(**base_model.get_params())

                m = evaluate_on_fixed_test(model, X_sub, y_sub, X_test, y_test)
                accs.append(m["accuracy"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                f1s.append(m["f1"])
                times.append(m["time"])

            results[model_name].append({
                "train_frac": frac,
                "n_train": n_sub,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "precision_mean": float(np.mean(precs)),
                "precision_std": float(np.std(precs)),
                "recall_mean": float(np.mean(recs)),
                "recall_std": float(np.std(recs)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "time_mean": float(np.mean(times)),
                "time_std": float(np.std(times)),
            })

    return results


def plot_exp5_curves(exp5_results, metric, outpath):
    plt.figure(figsize=(7, 5))

    for model_name, rows in exp5_results.items():
        xs = [r["n_train"] for r in rows]
        ys = [r[f"{metric}_mean"] for r in rows]
        yerr = [r[f"{metric}_std"] for r in rows]

        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=model_name)

    plt.xlabel("Taille du sous-ensemble d'entraînement (#lignes)")
    plt.ylabel(metric.upper())
    plt.title(f"Expérimentation 5 – {metric.upper()} vs taille entraînement (test fixe)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def evaluate_on_fixed_test(model, X_train_sub, y_train_sub, X_test, y_test):
    start = time.time()
    model.fit(X_train_sub, y_train_sub)
    train_time = time.time() - start

    # scores proba si dispo (utile AUC)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)
    metrics["time"] = train_time
    return metrics


def run_experiment_5(best_models, X_train, y_train, X_test, y_test,
                     train_fracs=None, repeats=3, random_state=42):

    if train_fracs is None:
        train_fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = {name: [] for name in best_models.keys()}

    n_total = len(X_train)

    for frac in train_fracs:
        n_sub = int(n_total * frac)

        # éviter les tailles trop petites
        if n_sub < 50:
            continue

        # ===============================
        # Cas spécial : frac == 1.0
        # ===============================
        if n_sub >= n_total:
            for model_name, base_model in best_models.items():
                model = base_model.__class__(**base_model.get_params())
                m = evaluate_on_fixed_test(model, X_train, y_train, X_test, y_test)

                results[model_name].append({
                    "train_frac": frac,
                    "n_train": n_total,
                    "accuracy_mean": m["accuracy"],
                    "accuracy_std": 0.0,
                    "precision_mean": m["precision"],
                    "precision_std": 0.0,
                    "recall_mean": m["recall"],
                    "recall_std": 0.0,
                    "f1_mean": m["f1"],
                    "f1_std": 0.0,
                    "time_mean": m["time"],
                    "time_std": 0.0,
                })
            continue

        # ===============================
        # Cas général : frac < 1.0
        # ===============================
        splitter = StratifiedShuffleSplit(
            n_splits=repeats,
            train_size=n_sub,
            random_state=random_state
        )

        for model_name, base_model in best_models.items():
            accs, precs, recs, f1s, times = [], [], [], [], []

            for train_idx, _ in splitter.split(X_train, y_train):
                X_sub = X_train.iloc[train_idx] if hasattr(X_train, "iloc") else X_train[train_idx]
                y_sub = y_train.iloc[train_idx] if hasattr(y_train, "iloc") else y_train[train_idx]

                model = base_model.__class__(**base_model.get_params())

                m = evaluate_on_fixed_test(model, X_sub, y_sub, X_test, y_test)

                accs.append(m["accuracy"])
                precs.append(m["precision"])
                recs.append(m["recall"])
                f1s.append(m["f1"])
                times.append(m["time"])

            results[model_name].append({
                "train_frac": frac,
                "n_train": n_sub,
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "precision_mean": float(np.mean(precs)),
                "precision_std": float(np.std(precs)),
                "recall_mean": float(np.mean(recs)),
                "recall_std": float(np.std(recs)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "time_mean": float(np.mean(times)),
                "time_std": float(np.std(times)),
            })

    return results



def plot_exp5_curves(exp5_results, metric, outpath):
    plt.figure(figsize=(7, 5))

    for model_name, rows in exp5_results.items():
        xs = [r["n_train"] for r in rows]
        ys = [r[f"{metric}_mean"] for r in rows]
        yerr = [r[f"{metric}_std"] for r in rows]

        plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=model_name)

    plt.xlabel("Taille du sous-ensemble d'entraînement (#lignes)")
    plt.ylabel(metric.upper())
    plt.title(f"Expérimentation 5 – {metric.upper()} vs taille entraînement (test fixe)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# =====================================================
# Main
# =====================================================

    def main():
        if len(sys.argv) != 4:
            print("Usage: python model_train_and_tune.py <features.csv> <labels.csv> <test_size>")
            sys.exit(1)

        features_path = sys.argv[1]
        labels_path = sys.argv[2]
        test_size = float(sys.argv[3])

        print(" Loading data...")
        X, y = read_features_and_labels(features_path, labels_path)

        print(" Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True
        )

        print("Train shape:", X_train.shape)
        print("Test shape :", X_test.shape)

        # =================================================
        # EXPÉRIMENTATION 1
        # =================================================
        print("\n=== EXPÉRIMENTATION 1 : Modèles par défaut ===")

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
            metrics_test, t = train_and_evaluate_default(model, name, X_train, y_train, X_test, y_test)
            print(f"{name} | Accuracy={metrics_test['accuracy']:.4f} | Precision={metrics_test['precision']:.4f}")

        # =================================================
        # EXPÉRIMENTATION 2
        # =================================================
        print("\n=== EXPÉRIMENTATION 2 : GridSearchCV ===")

        param_grids = {
            "RandomForest": (
                RandomForestClassifier(random_state=42),
                {"n_estimators": [100, 300], "max_depth": [None, 10, 20]}
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

        exp3_results = {}
        for name, (model, grid) in param_grids.items():
            exp3_results[name] = run_grid_search_and_evaluate(model, grid, name, X_train, y_train, X_test, y_test)

        # =================================================
        # EXPÉRIMENTATION 3
        # =================================================
        print("\n=== EXPÉRIMENTATION 3 : Comparaison des meilleurs modèles ===")

        for metric in ["accuracy", "precision", "recall", "f1"]:
            plot_metric_comparison(exp3_results, metric)

        # Save summary table (Expé 3)
        rows = []
        for name, res in exp3_results.items():
            rows.append({
                "Model": name,
                "Accuracy_train": res["train"]["accuracy"],
                "Accuracy_test": res["test"]["accuracy"],
                "Precision_test": res["test"]["precision"],
                "Recall_test": res["test"]["recall"],
                "F1_test": res["test"]["f1"],
                "Time_s": res["time"]
            })
        pd.DataFrame(rows).to_csv(f"{EXP3_DIR}/exp3_summary_metrics.csv", index=False)
        print(" Résultats Expérimentation 3 sauvegardés dans tests/exp3/")

        # =================================================
        # EXPÉRIMENTATION 4
        # =================================================
        print("\n=== EXPÉRIMENTATION 4 : Inférence sur d'autres jeux de données ===")

        external_datasets = {
            "CO": (
                "../Complementary-data/acsincome_co_allfeatures.csv",
                "../Complementary-data/acsincome_co_label.csv"
            ),
            "NE": (
                "../Complementary-data/acsincome_ne_allfeatures.csv",
                "../Complementary-data/acsincome_ne_label.csv"
            )
        }

        exp4_results = {}

        for state, (X_path, y_path) in external_datasets.items():
            print(f"\n--- Inférence sur l'État {state} ---")

            X_ext = pd.read_csv(X_path)
            y_ext = pd.read_csv(y_path).iloc[:, 0]

            exp4_results[state] = {}

            for model_name, res in exp3_results.items():
                metrics = inference_on_new_dataset(
                    model=res["best_model"],
                    model_name=model_name,
                    dataset_name=state,
                    X_new=X_ext,
                    y_new=y_ext,
                    output_dir=f"{EXP4_DIR}/{state}"
                )

                exp4_results[state][model_name] = metrics

                print(
                    f"{model_name} | "
                    f"Acc={metrics['accuracy']:.4f} | "
                    f"Prec={metrics['precision']:.4f} | "
                    f"Rec={metrics['recall']:.4f} | "
                    f"F1={metrics['f1']:.4f} | "
                    f"AUC={metrics['auc']:.4f}"
                )
        # =================================================
        # EXPÉRIMENTATION 5 – impact taille dataset
        # =================================================
        print("\n=== EXPÉRIMENTATION 5 : Impact de la taille des données ===")

        # On récupère les meilleurs modèles (hyperparams optimaux)
        best_models = {
            name: res["best_model"]
            for name, res in exp3_results.items()
        }

        exp5_results = run_experiment_5(
            best_models=best_models,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            train_fracs=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            repeats=3,
            random_state=42
        )

        # Sauvegarde CSV
        for model_name, rows in exp5_results.items():
            pd.DataFrame(rows).to_csv(f"{EXP5_DIR}/exp5_{model_name.lower()}.csv", index=False)

        # Courbes
        plot_exp5_curves(exp5_results, "accuracy",  f"{EXP5_DIR}/accuracy_vs_train_size.png")
        plot_exp5_curves(exp5_results, "precision", f"{EXP5_DIR}/precision_vs_train_size.png")
        plot_exp5_curves(exp5_results, "recall",    f"{EXP5_DIR}/recall_vs_train_size.png")
        plot_exp5_curves(exp5_results, "f1",        f"{EXP5_DIR}/f1_vs_train_size.png")

        print(f" Expé 5 terminée. Courbes et CSV sauvegardés dans {EXP5_DIR}/")



    if __name__ == "__main__":
        main()