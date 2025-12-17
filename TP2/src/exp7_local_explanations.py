import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- LIME ---
from lime.lime_tabular import LimeTabularExplainer

# --- SHAP (plots only) ---
import shap

# --- XGBoost ---
import xgboost as xgb
from xgboost import XGBClassifier


# =====================================================
# Config
# =====================================================
FEATURES_PATH = "../Dataset/alt_acsincome_ca_features_85.csv"
LABELS_PATH = "../Dataset/alt_acsincome_ca_labels_85.csv"

OUTDIR = "tests/exp7"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f"{OUTDIR}/lime", exist_ok=True)
os.makedirs(f"{OUTDIR}/shap", exist_ok=True)
os.makedirs(f"{OUTDIR}/shap/groups", exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Local explanations (LIME + SHAP waterfall)
N_LOCAL_EXAMPLES = 6

# For SHAP summary plots (speed/memory)
MAX_SUMMARY_SAMPLES = 10000
MAX_GROUP_SAMPLES = 5000


# =====================================================
# Load + split
# =====================================================
X = pd.read_csv(FEATURES_PATH)
y = pd.read_csv(LABELS_PATH).iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, shuffle=True, stratify=y, random_state=RANDOM_STATE
)

feature_names = list(X.columns)
class_names = ["<=50K", ">50K"]


# =====================================================
# Train best model (mets tes meilleurs hyperparams Exp2)
# =====================================================
model = XGBClassifier(
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# Predictions + confusion groups
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)


# =====================================================
# Helper: pick examples
# =====================================================
def pick_example_indices(X_test, y_test, y_pred, n_each=2):
    """Return indices for TP, TN, FP, FN (n_each each) if possible."""
    idx = X_test.index

    tp = idx[(y_test == 1) & (y_pred == 1)]
    tn = idx[(y_test == 0) & (y_pred == 0)]
    fp = idx[(y_test == 0) & (y_pred == 1)]
    fn = idx[(y_test == 1) & (y_pred == 0)]

    def take(arr):
        return list(arr[:n_each]) if len(arr) >= n_each else list(arr)

    return {
        "TP": take(tp),
        "TN": take(tn),
        "FP": take(fp),
        "FN": take(fn),
    }


# =====================================================
# LIME
# =====================================================
def run_lime(X_train, X_test, model, outdir, example_indices):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True
    )

    for k, idx in enumerate(example_indices):
        x = X_test.loc[idx].values

        exp = explainer.explain_instance(
            data_row=x,
            predict_fn=model.predict_proba,
            num_features=10
        )

        # === PNG ONLY (no HTML) ===
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        png_path = f"{outdir}/lime/lime_explanation_{k}_idx{idx}.png"
        plt.savefig(png_path, dpi=200)
        plt.close()

        print(f"[LIME] saved {png_path}")



# =====================================================
# SHAP values using XGBoost native pred_contribs=True
# (avoids shap.TreeExplainer crash)
# =====================================================
def xgb_shap_contribs(model: XGBClassifier, X_df: pd.DataFrame):
    """
    Returns:
      shap_vals: (n_samples, n_features) contributions
      base_vals: (n_samples,) bias term
    Note: values are in log-odds (margin) space for binary logistic.
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_df.values, feature_names=list(X_df.columns))
    contrib = booster.predict(dmat, pred_contribs=True)  # (n, p+1)
    shap_vals = contrib[:, :-1]
    base_vals = contrib[:, -1]
    return shap_vals, base_vals


# =====================================================
# SHAP Waterfall (local)
# =====================================================
def run_shap_waterfall(X_test, model, outdir, example_indices):
    X_sel = X_test.loc[example_indices]
    shap_vals, base_vals = xgb_shap_contribs(model, X_sel)

    for i, idx in enumerate(example_indices):
        exp_i = shap.Explanation(
            values=shap_vals[i],
            base_values=base_vals[i],
            data=X_sel.loc[idx].values,
            feature_names=X_sel.columns
        )

        plt.figure()
        shap.plots.waterfall(exp_i, max_display=12, show=False)
        plt.tight_layout()
        out_png = f"{outdir}/shap/waterfall_{i}_idx{idx}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[SHAP] saved {out_png}")


# =====================================================
# SHAP summary_plot (global)
# =====================================================
def run_shap_summary(X_test, model, outdir):
    if len(X_test) > MAX_SUMMARY_SAMPLES:
        X_small = X_test.sample(n=MAX_SUMMARY_SAMPLES, random_state=RANDOM_STATE)
    else:
        X_small = X_test

    shap_vals, _ = xgb_shap_contribs(model, X_small)

    plt.figure()
    shap.summary_plot(
        shap_vals,
        X_small,
        show=False,
        max_display=15
    )
    plt.tight_layout()
    out_png = f"{outdir}/shap/summary_plot_top15.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[SHAP] saved {out_png}")


# =====================================================
# SHAP summary_plot by TP/TN/FP/FN
# =====================================================
def run_shap_summary_by_groups(X_test, y_test, y_pred, model, outdir):
    groups = {
        "TP": X_test[(y_test == 1) & (y_pred == 1)],
        "TN": X_test[(y_test == 0) & (y_pred == 0)],
        "FP": X_test[(y_test == 0) & (y_pred == 1)],
        "FN": X_test[(y_test == 1) & (y_pred == 0)],
    }

    for gname, Xg in groups.items():
        if len(Xg) == 0:
            print(f"[SHAP] Group {gname} empty -> skip")
            continue

        if len(Xg) > MAX_GROUP_SAMPLES:
            Xg_small = Xg.sample(n=MAX_GROUP_SAMPLES, random_state=RANDOM_STATE)
        else:
            Xg_small = Xg

        shap_vals_g, _ = xgb_shap_contribs(model, Xg_small)

        plt.figure()
        shap.summary_plot(
            shap_vals_g,
            Xg_small,
            show=False,
            max_display=15
        )
        plt.tight_layout()
        out_png = f"{outdir}/shap/groups/summary_{gname}_top15.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"[SHAP] saved {out_png}")


# =====================================================
# Main execution
# =====================================================
if __name__ == "__main__":
    group_idxs = pick_example_indices(X_test, y_test, y_pred, n_each=2)
    print("Picked indices:", group_idxs)

    local_indices = []
    for g, ids in group_idxs.items():
        local_indices += ids
    local_indices = local_indices[:N_LOCAL_EXAMPLES]

    # LIME local
    run_lime(X_train, X_test, model, OUTDIR, local_indices)

    # SHAP local waterfall (XGBoost native SHAP values)
    run_shap_waterfall(X_test, model, OUTDIR, local_indices)

    # SHAP global summary
    run_shap_summary(X_test, model, OUTDIR)

    # SHAP grouped TP/TN/FP/FN summaries
    run_shap_summary_by_groups(X_test, y_test, y_pred, model, OUTDIR)

    print("\n Exp7 terminé. Résultats dans tests/exp7/")
