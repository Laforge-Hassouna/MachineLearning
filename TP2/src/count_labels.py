import pandas as pd
import os
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 count_labels.py <labels.csv>")
        sys.exit(1)

    labels_path = sys.argv[1]
    y = pd.read_csv(labels_path)

    # si une seule colonne -> Series
    if y.shape[1] == 1:
        y = y.iloc[:, 0]
    else:
        # sinon on prend la première colonne (au cas où)
        y = y.iloc[:, 0]

    # normalisation (True/False, 1/0, "true"/"false", etc.)
    y_norm = y.astype(str).str.strip().str.lower()
    y_norm = y_norm.replace({
        "true": 1, "false": 0,
        "1": 1, "0": 0
    })

    # conversion finale en int (si ça échoue, c'est que le fichier a un format inattendu)
    y_norm = y_norm.astype(int)

    counts = y_norm.value_counts().sort_index()   # 0 puis 1
    n_false = int(counts.get(0, 0))
    n_true  = int(counts.get(1, 0))
    n_total = int(len(y_norm))

    print(f"False (0) = {n_false}")
    print(f"True  (1) = {n_true}")
    print(f"Total     = {n_total}")

    out_dir = "tests"
    os.makedirs(out_dir, exist_ok=True)

    out = pd.DataFrame([{
        "False(0)": n_false,
        "True(1)": n_true,
        "Total": n_total
    }])

    out_path = os.path.join(out_dir, "label_counts.csv")
    out.to_csv(out_path, index=False)
    print(f" Saved: {out_path}")

if __name__ == "__main__":
    main()
