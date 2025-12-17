#!/usr/bin/env python3
# exp5_plots.py
# Analyse + plots pour l'Expérimentation 5 (impact de la taille des données)

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["accuracy", "precision", "recall", "f1", "time"]


def load_exp5_csvs(exp5_dir: str, files: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """
    Charge les CSV exp5 par modèle.
    Par défaut, cherche exp5_*.csv dans exp5_dir.
    Retour: dict {model_name: dataframe}
    """
    if files is None or len(files) == 0:
        pattern = os.path.join(exp5_dir, "exp5_*.csv")
        files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Aucun fichier exp5_*.csv trouvé.\n"
            f"Vérifie le dossier: {exp5_dir}\n"
            f"Ou passe explicitement les fichiers avec --files ..."
        )

    data = {}
    for f in files:
        df = pd.read_csv(f)
        base = os.path.basename(f).replace(".csv", "")
        # exp5_randomforest -> RandomForest
        model_name = base.replace("exp5_", "").replace("_", " ").title().replace(" ", "")
        data[model_name] = df

    return data


def sanity_check(df: pd.DataFrame, model_name: str):
    required = [
        "train_frac", "n_train",
        "accuracy_mean", "accuracy_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "time_mean", "time_std"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{model_name}] colonnes manquantes dans le CSV: {missing}")


def plot_metric_vs_train_size(
    data: dict[str, pd.DataFrame],
    metric: str,
    out_dir: str,
    show: bool = False
):
    """
    Courbe (avec barres d'erreur) de la métrique en fonction de n_train.
    """
    plt.figure(figsize=(8, 5))

    for model_name, df in data.items():
        x = df["n_train"].tolist()

        y_col = f"{metric}_mean"
        e_col = f"{metric}_std"

        y = df[y_col].tolist()
        yerr = df[e_col].tolist()

        plt.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=model_name)

    plt.xlabel("Taille du sous-ensemble d'entraînement (#lignes)")
    plt.ylabel(metric.upper() if metric != "time" else "TIME (s)")
    plt.title(f"Expérimentation 5 – {metric.upper()} vs taille d'entraînement (test fixe)")
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"{metric}_vs_train_size.png")
    plt.savefig(outpath)
    if show:
        plt.show()
    plt.close()

    print(f" Plot sauvegardé: {outpath}")


def plot_final_comparison_bar(
    data: dict[str, pd.DataFrame],
    metric: str,
    out_dir: str,
    show: bool = False
):
    """
    Bar plot: comparaison des modèles sur la dernière taille (max n_train) pour une métrique.
    """
    rows = []
    for model_name, df in data.items():
        df_sorted = df.sort_values("n_train")
        last = df_sorted.iloc[-1]
        rows.append((model_name, float(last[f"{metric}_mean"]), float(last[f"{metric}_std"]), int(last["n_train"])))

    rows.sort(key=lambda x: x[1], reverse=True)

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    errs = [r[2] for r in rows]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values, yerr=errs, capsize=4)
    plt.ylabel(metric.upper() if metric != "time" else "TIME (s)")
    plt.title(f"Expé 5 – Comparaison finale ({metric.upper()}) sur n_train maximal")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"final_{metric}_comparison.png")
    plt.savefig(outpath)
    if show:
        plt.show()
    plt.close()

    print(f" Plot sauvegardé: {outpath}")

    # Petit résumé terminal
    max_n = rows[0][3] if rows else None
    print(f"\n--- Comparaison finale ({metric}) (n_train≈{max_n}) ---")
    for m, v, e, n in rows:
        print(f"{m:12s}  {metric}={v:.4f}  ±{e:.4f}   (n_train={n})")


def main():
    parser = argparse.ArgumentParser(description="Plots Expérimentation 5 (impact taille dataset)")
    parser.add_argument("--exp5_dir", default="tests/exp5", help="Dossier contenant exp5_*.csv")
    parser.add_argument("--out_dir", default="tests/exp5/plots", help="Dossier où sauvegarder les plots")
    parser.add_argument("--show", action="store_true", help="Afficher les figures à l'écran (optionnel)")
    parser.add_argument("--files", nargs="*", help="Liste de CSV à charger (optionnel)")
    args = parser.parse_args()

    data = load_exp5_csvs(args.exp5_dir, args.files)

    # Sanity check
    for model_name, df in data.items():
        sanity_check(df, model_name)

    # 1) Courbes métriques vs taille
    for metric in METRICS:
        plot_metric_vs_train_size(data, metric, args.out_dir, show=args.show)

    # 2) Comparaisons finales (barplots)
    for metric in METRICS:
        plot_final_comparison_bar(data, metric, args.out_dir, show=args.show)

    print("\nTerminé. Tous les plots sont dans:", args.out_dir)


if __name__ == "__main__":
    main()
