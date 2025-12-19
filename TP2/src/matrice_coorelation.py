import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================================================
# Paths
# =====================================================
FEATURES_PATH = "../Dataset/alt_acsincome_ca_features_85.csv"
LABELS_PATH   = "../Dataset/alt_acsincome_ca_labels_85.csv"

OUTDIR = "tests/correlation"
os.makedirs(OUTDIR, exist_ok=True)

# =====================================================
# Load data
# =====================================================
X = pd.read_csv(FEATURES_PATH)
y = pd.read_csv(LABELS_PATH).iloc[:, 0]

# Ajouter le label aux features
df = X.copy()
df["income_>50K"] = y

# =====================================================
# Correlation matrix (Pearson)
# =====================================================
corr = df.corr(method="pearson")

# Sauvegarde CSV
corr.to_csv(f"{OUTDIR}/correlation_matrix.csv")

# =====================================================
# Heatmap (lisible : top variables seulement)
# =====================================================
# On sélectionne les variables les plus corrélées au label
top_features = (
    corr["income_>50K"]
    .abs()
    .sort_values(ascending=False)
    .head(15)
    .index
)

corr_top = corr.loc[top_features, top_features]

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_top,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Matrice de corrélation – Top variables (California)")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/correlation_heatmap_top15.png", dpi=200)
plt.close()

print("Matrice de corrélation sauvegardée dans :", OUTDIR)
