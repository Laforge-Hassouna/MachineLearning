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
EXP8_DIR = "tests/exp8"

for d in [OUTPUT_DIR, EXP2_DIR, EXP3_DIR, EXP4_DIR, EXP5_DIR, EXP8_DIR]:
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
    cv=5
):
    print(f"\n GridSearchCV – {model_name}")

    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=2,  # stable sous WSL
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

def save_exp3_summary(exp3_results):
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
    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP3_DIR}/exp3_summary_metrics.csv", index=False)
    print(" Résultats Expérimentation 3 sauvegardés dans tests/exp3/")


# =====================================================
# Expérimentation 4 – inference + curves
# =====================================================

def inference_on_new_dataset(model, model_name, dataset_name, X_new, y_new, output_dir):
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_new)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_new)
        # normalisation simple si besoin (fallback)
        y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)
    else:
        y_score = model.predict(X_new)

    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_new, y_pred),
        "precision": precision_score(y_new, y_pred, zero_division=0),
        "recall": recall_score(y_new, y_pred, zero_division=0),
        "f1": f1_score(y_new, y_pred, zero_division=0),
        "cm": confusion_matrix(y_new, y_pred)
    }

    print_confusion_matrix(metrics["cm"], model_name, dataset_name, "Inference")

    save_confusion_matrix(
        metrics["cm"],
        f"{model_name} – Confusion Matrix (Inference) – {dataset_name}",
        f"{output_dir}/confusion_matrix_{model_name.lower()}.png"
    )

    metrics["auc"] = plot_roc_curve(
        y_new, y_score,
        title=f"ROC – {model_name} – {dataset_name}",
        filepath=f"{output_dir}/roc_{model_name.lower()}.png"
    )

    plot_pr_curve(
        y_new, y_score,
        title=f"Precision–Recall – {model_name} – {dataset_name}",
        filepath=f"{output_dir}/pr_{model_name.lower()}.png"
    )

    plot_threshold_metrics(
        y_new, y_score,
        title=f"Threshold vs Metrics – {model_name} – {dataset_name}",
        filepath=f"{output_dir}/threshold_metrics_{model_name.lower()}.png"
    )

    return metrics


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
                     train_fracs=None, repeats=3, random_state=42):

    if train_fracs is None:
        train_fracs = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    results = {name: [] for name in best_models.keys()}
    n_total = len(X_train)

    for frac in train_fracs:
        n_sub = int(n_total * frac)
        if n_sub < 50:
            continue

        # cas frac == 1.0
        if n_sub >= n_total:
            for model_name, base_model in best_models.items():
                model = base_model.__class__(**base_model.get_params())
                m = evaluate_on_fixed_test(model, X_train, y_train, X_test, y_test)

                results[model_name].append({
                    "train_frac": frac,
                    "n_train": n_total,
                    "accuracy_mean": m["accuracy"], "accuracy_std": 0.0,
                    "precision_mean": m["precision"], "precision_std": 0.0,
                    "recall_mean": m["recall"], "recall_std": 0.0,
                    "f1_mean": m["f1"], "f1_std": 0.0,
                    "time_mean": m["time"], "time_std": 0.0,
                })
            continue

        splitter = StratifiedShuffleSplit(
            n_splits=repeats,
            train_size=n_sub,
            random_state=random_state
        )

        for model_name, base_model in best_models.items():
            accs, precs, recs, f1s, times = [], [], [], [], []

            for train_idx, _ in splitter.split(X_train, y_train):
                X_sub = X_train.iloc[train_idx]
                y_sub = y_train.iloc[train_idx]

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
                "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                "precision_mean": float(np.mean(precs)), "precision_std": float(np.std(precs)),
                "recall_mean": float(np.mean(recs)), "recall_std": float(np.std(recs)),
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "time_mean": float(np.mean(times)), "time_std": float(np.std(times)),
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
# EXPÉRIMENTATION 8 – Explication contrefactuelle
# =====================================================

def predict_scores_proba(model, Xdf):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xdf)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(Xdf)
        # normalisation simple en [0,1]
        return (s - s.min()) / (s.max() - s.min() + 1e-12)
    # fallback (pas idéal)
    return model.predict(Xdf).astype(float)

def generate_grid_from_train(X_train, feature, x0, n_points=35):
    vals = X_train[feature].values
    qs = np.linspace(0.01, 0.99, n_points)
    grid = np.quantile(vals, qs)
    grid = np.unique(np.append(grid, x0))
    return grid

def one_feature_whatif(model, X_train, X_test, idx, feature, outdir):
    x_orig = X_test.loc[idx].copy()
    base_proba = float(predict_scores_proba(model, pd.DataFrame([x_orig]))[0])
    base_pred = int(base_proba >= 0.5)

    grid = generate_grid_from_train(X_train, feature, x_orig[feature], n_points=35)
    probs = []
    for v in grid:
        x_new = x_orig.copy()
        x_new[feature] = v
        probs.append(float(predict_scores_proba(model, pd.DataFrame([x_new]))[0]))

    # plot
    plt.figure(figsize=(7, 4))
    plt.plot(grid, probs)
    plt.axhline(0.5, linestyle="--")
    plt.scatter([x_orig[feature]], [base_proba], marker="o")
    plt.xlabel(feature)
    plt.ylabel("P(>50K)")
    plt.title(f"What-if – idx={idx} – base_pred={base_pred}")
    plt.tight_layout()

    fname = f"{outdir}/whatif_idx{idx}_{feature}.png"
    plt.savefig(fname)
    plt.close()

    # minimal flip (single feature)
    target = 1 - base_pred
    best = None
    for v, p in zip(grid, probs):
        pred = int(p >= 0.5)
        if pred == target:
            cost = abs(v - x_orig[feature])
            if (best is None) or (cost < best["delta_abs"]):
                best = {"new_value": v, "new_proba": p, "delta_abs": cost}

    return base_pred, base_proba, best, fname

def pick_tp_tn_fp_fn_indices(model, X_test, y_test, n_each=1, seed=42):
    rng = np.random.default_rng(seed)
    scores = predict_scores_proba(model, X_test)
    y_pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print("\n[EXP8] Confusion matrix (tuned model on test):")
    print(cm)

    idx = X_test.index
    TP = idx[(y_test == 1) & (y_pred == 1)].to_numpy()
    TN = idx[(y_test == 0) & (y_pred == 0)].to_numpy()
    FP = idx[(y_test == 0) & (y_pred == 1)].to_numpy()
    FN = idx[(y_test == 1) & (y_pred == 0)].to_numpy()

    def sample(arr):
        if len(arr) == 0:
            return []
        k = min(n_each, len(arr))
        return rng.choice(arr, size=k, replace=False).tolist()

    return {"TP": sample(TP), "TN": sample(TN), "FP": sample(FP), "FN": sample(FN)}

def run_experiment_8_counterfactuals(best_model, X_train, X_test, y_test):
    print("\n=== EXPÉRIMENTATION 8 : Explication contrefactuelle (what-if) ===")
    os.makedirs(EXP8_DIR, exist_ok=True)

    # on prend 1 exemple TP/TN/FP/FN si possible
    picked = pick_tp_tn_fp_fn_indices(best_model, X_test, y_test, n_each=1, seed=42)
    print("[EXP8] Picked indices:", picked)

    # features à tester
    features_to_test = ["WKHP", "SCHL", "AGEP", "OCCP", "RELP", "SEX", "MAR", "COW", "POBP", "RAC1P"]

    rows = []
    for group, indices in picked.items():
        for idx in indices:
            for f in features_to_test:
                base_pred, base_proba, best, plotfile = one_feature_whatif(
                    best_model, X_train, X_test, idx, f, EXP8_DIR
                )
                rows.append({
                    "group": group,
                    "idx": idx,
                    "feature": f,
                    "base_pred": base_pred,
                    "base_proba": base_proba,
                    "can_flip": (best is not None),
                    "new_value": (best["new_value"] if best else np.nan),
                    "new_proba": (best["new_proba"] if best else np.nan),
                    "delta_abs": (best["delta_abs"] if best else np.nan),
                    "plot": plotfile
                })

    df = pd.DataFrame(rows)
    df.to_csv(f"{EXP8_DIR}/counterfactual_summary.csv", index=False)

    # résumé minimal : meilleur attribut par individu
    best_per_idx = (
        df[df["can_flip"]]
        .sort_values(["idx", "delta_abs"])
        .groupby("idx")
        .head(1)
    )
    best_per_idx.to_csv(f"{EXP8_DIR}/counterfactual_best_single_feature.csv", index=False)

    print(f"[EXP8] Saved: {EXP8_DIR}/counterfactual_summary.csv")
    print(f"[EXP8] Saved: {EXP8_DIR}/counterfactual_best_single_feature.csv")
    if len(best_per_idx) > 0:
        print("\n[EXP8] Best single-feature flips (first rows):")
        print(best_per_idx[["group", "idx", "feature", "base_pred", "base_proba", "new_value", "new_proba", "delta_abs"]].head(10))
    else:
        print("[EXP8] Aucun flip trouvé en modifiant une seule feature (rare mais possible).")


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
        metrics_test, _ = train_and_evaluate_default(model, name, X_train, y_train, X_test, y_test)
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
        exp3_results[name] = run_grid_search_and_evaluate(model, grid, name, X_train, y_train, X_test, y_test, cv=5)

    # =================================================
    # EXPÉRIMENTATION 3
    # =================================================
    print("\n=== EXPÉRIMENTATION 3 : Comparaison des meilleurs modèles ===")

    for metric in ["accuracy", "precision", "recall", "f1"]:
        plot_metric_comparison(exp3_results, metric)

    save_exp3_summary(exp3_results)

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

    for state, (X_path, y_path) in external_datasets.items():
        print(f"\n--- Inférence sur l'État {state} ---")

        X_ext = pd.read_csv(X_path)
        y_ext = pd.read_csv(y_path).iloc[:, 0]

        for model_name, res in exp3_results.items():
            metrics = inference_on_new_dataset(
                model=res["best_model"],
                model_name=model_name,
                dataset_name=state,
                X_new=X_ext,
                y_new=y_ext,
                output_dir=f"{EXP4_DIR}/{state}"
            )

            print(
                f"{model_name} | "
                f"Acc={metrics['accuracy']:.4f} | "
                f"Prec={metrics['precision']:.4f} | "
                f"Rec={metrics['recall']:.4f} | "
                f"F1={metrics['f1']:.4f} | "
                f"AUC={metrics['auc']:.4f}"
            )

    # =================================================
    # EXPÉRIMENTATION 5
    # =================================================
    print("\n=== EXPÉRIMENTATION 5 : Impact de la taille des données ===")

    best_models = {name: res["best_model"] for name, res in exp3_results.items()}

    exp5_results = run_experiment_5(
        best_models=best_models,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        train_fracs=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        repeats=3,
        random_state=42
    )

    for model_name, rows in exp5_results.items():
        pd.DataFrame(rows).to_csv(f"{EXP5_DIR}/exp5_{model_name.lower()}.csv", index=False)

    plot_exp5_curves(exp5_results, "accuracy",  f"{EXP5_DIR}/accuracy_vs_train_size.png")
    plot_exp5_curves(exp5_results, "precision", f"{EXP5_DIR}/precision_vs_train_size.png")
    plot_exp5_curves(exp5_results, "recall",    f"{EXP5_DIR}/recall_vs_train_size.png")
    plot_exp5_curves(exp5_results, "f1",        f"{EXP5_DIR}/f1_vs_train_size.png")

    print(f" Expé 5 terminée. Courbes et CSV sauvegardés dans {EXP5_DIR}/")

    # =================================================
    # EXPÉRIMENTATION 8 (Contrefactuel)
    # =================================================
    # On prend le meilleur modèle (celui avec meilleure accuracy test en exp3)
    best_name = max(exp3_results.keys(), key=lambda k: exp3_results[k]["test"]["accuracy"])
    tuned_best_model = exp3_results[best_name]["best_model"]
    print(f"\n[EXP8] Modèle sélectionné pour contrefactuel: {best_name}")

    run_experiment_8_counterfactuals(tuned_best_model, X_train, X_test, y_test)


if __name__ == "__main__":
    main()
