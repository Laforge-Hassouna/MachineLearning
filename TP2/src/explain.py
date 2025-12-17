import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


# =====================================================
# Config
# =====================================================
EXP6_DIR = "tests/exp6"
os.makedirs(EXP6_DIR, exist_ok=True)

FEATURES_PATH = "../Dataset/alt_acsincome_ca_features_85.csv"
LABELS_PATH = "../Dataset/alt_acsincome_ca_labels_85.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25

# IMPORTANT: to avoid joblib memory crashes
N_REPEATS = 5       # 5 is enough for a TP
N_JOBS = 1          # avoid parallel workers killed by OS
SAMPLE_TEST = 10000 # evaluate permutation importance on a subset of test
TOPK = 20


# =====================================================
# Helpers
# =====================================================
def save_topk_plot(importances_df, model_name, outdir, topk=20):
    top = importances_df.head(topk).iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Baisse moyenne du F1 après permutation")
    plt.title(f"Permutation Feature Importance (Top {topk}) – {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "perm_importance_top20.png"))
    plt.close()


def save_grouped_plot(importances_df, model_name, outdir, topk=15):
    tmp = importances_df.copy()
    tmp["group"] = tmp["feature"].apply(lambda s: str(s).split("_")[0])
    grouped = tmp.groupby("group")["importance_mean"].sum().sort_values(ascending=False).head(topk)

    plt.figure(figsize=(8, 5))
    grouped.iloc[::-1].plot(kind="barh")
    plt.xlabel("Somme des importances (groupées)")
    plt.title(f"Permutation Importance groupée (Top {topk}) – {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "perm_importance_grouped_top15.png"))
    plt.close()


def run_perm_importance(model, model_name, X_train, y_train, X_test, y_test):
    outdir = os.path.join(EXP6_DIR, model_name.lower())
    os.makedirs(outdir, exist_ok=True)

    # train
    model.fit(X_train, y_train)

    # baseline F1
    y_pred = model.predict(X_test)
    baseline_f1 = f1_score(y_test, y_pred)
    print(f"\n[{model_name}] Baseline F1 (test) = {baseline_f1:.4f}")

    # --- subset test to avoid memory issues ---
    if len(X_test) > SAMPLE_TEST:
        X_eval = X_test.sample(n=SAMPLE_TEST, random_state=RANDOM_STATE)
        y_eval = y_test.loc[X_eval.index]
        print(f"[{model_name}] Permutation importance computed on a subset: {SAMPLE_TEST} rows")
    else:
        X_eval = X_test
        y_eval = y_test

    # permutation importance (robust settings)
    perm = permutation_importance(
        model,
        X_eval,
        y_eval,
        scoring="f1",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    importances = pd.DataFrame({
        "feature": X_eval.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    # save CSV
    importances.to_csv(os.path.join(outdir, "perm_importance_all.csv"), index=False)

    # plots
    save_topk_plot(importances, model_name, outdir, topk=TOPK)
    save_grouped_plot(importances, model_name, outdir, topk=15)

    print(f"[{model_name}] Saved: {outdir}/perm_importance_all.csv")
    print(f"[{model_name}] Saved: {outdir}/perm_importance_top20.png")
    print(f"[{model_name}] Saved: {outdir}/perm_importance_grouped_top15.png")


# =====================================================
# Main
# =====================================================
def main():
    X = pd.read_csv(FEATURES_PATH)
    y = pd.read_csv(LABELS_PATH).iloc[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Best models (use your Exp2 best params)
    best_rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=300,
        max_depth=20
    )

    best_ada = AdaBoostClassifier(
        random_state=RANDOM_STATE,
        n_estimators=200,
        learning_rate=1.0
    )

    best_xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1
    )

    models = {
        "RandomForest": best_rf,
        "AdaBoost": best_ada,
        "XGBoost": best_xgb
    }

    print("=== EXP6 : Permutation Feature Importance ===")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Scoring: F1 | n_repeats={N_REPEATS} | n_jobs={N_JOBS}")

    for name, model in models.items():
        run_perm_importance(model, name, X_train, y_train, X_test, y_test)

    print("\nExp6 terminé. Résultats dans tests/exp6/")


if __name__ == "__main__":
    main()
