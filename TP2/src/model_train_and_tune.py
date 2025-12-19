import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from xgboost import XGBClassifier
import xgboost as xgb

# LIME
from lime.lime_tabular import LimeTabularExplainer

# SHAP
import shap


# =====================================================
# Paths
# =====================================================
TRAIN_FEATURES = "../Dataset/alt_acsincome_ca_features_85.csv"
TRAIN_LABELS   = "../Dataset/alt_acsincome_ca_labels_85.csv"

TEST_FEATURES  = "../Dataset/features_split_5.csv"
TEST_LABELS    = "../Dataset/labels_split_5.csv"

OUTDIR = "tests/split5_xgb"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(f"{OUTDIR}/lime", exist_ok=True)
os.makedirs(f"{OUTDIR}/shap", exist_ok=True)


# =====================================================
# Load data
# =====================================================
X_train_full = pd.read_csv(TRAIN_FEATURES)
y_train_full = pd.read_csv(TRAIN_LABELS).iloc[:, 0]

X_test_new = pd.read_csv(TEST_FEATURES)
y_test_new = pd.read_csv(TEST_LABELS).iloc[:, 0]


# =====================================================
# Train / validation split (internal)
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)


# =====================================================
# Best XGBoost model (from Exp2)
# =====================================================
model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1
)

model.fit(X_train, y_train)


# =====================================================
# Evaluation on NEW dataset (split 5)
# =====================================================
y_pred_new = model.predict(X_test_new)
acc = accuracy_score(y_test_new, y_pred_new)
cm = confusion_matrix(y_test_new, y_pred_new)

print("\n=== Evaluation on NEW test dataset (split 5) ===")
print(f"Accuracy = {acc:.4f}")
print("Confusion matrix:\n", cm)


# =====================================================
# LIME – local explanations (3 samples)
# =====================================================
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=["<=50K", ">50K"],
    mode="classification",
    discretize_continuous=True
)

for i, idx in enumerate(X_test_new.index[:3]):
    exp = explainer.explain_instance(
        X_test_new.loc[idx].values,
        model.predict_proba,
        num_features=10
    )

    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/lime/lime_{i}.png", dpi=200)
    plt.close()


# =====================================================
# SHAP – native XGBoost contributions
# =====================================================
def xgb_shap_contribs(model, X):
    booster = model.get_booster()
    dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    return contribs[:, :-1], contribs[:, -1]


# --- Global SHAP summary ---
shap_vals, _ = xgb_shap_contribs(model, X_test_new)

plt.figure()
shap.summary_plot(shap_vals, X_test_new, show=False, max_display=15)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/shap/summary.png", dpi=200)
plt.close()


# --- Local SHAP waterfall (1 sample) ---
exp = shap.Explanation(
    values=shap_vals[0],
    base_values=0,
    data=X_test_new.iloc[0].values,
    feature_names=X_test_new.columns
)

plt.figure()
shap.plots.waterfall(exp, show=False, max_display=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/shap/waterfall_0.png", dpi=200)
plt.close()


print("\nAnalyse terminée. Résultats dans :", OUTDIR)
