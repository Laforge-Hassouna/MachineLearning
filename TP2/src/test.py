import pandas as pd
import sys

import preproc as pp


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        sys.exit(1)

    print("Feature names before binarization:")
    print(df.columns.tolist())

    # Specify which columns to binarize (change as needed)
    columns_to_bin = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P']  # Replace with your actual column names

    df_binarized = pp.binarize_features(df, columns_to_bin)

    print("\nFeature names after binarization:")
    print(df_binarized.columns.tolist())

    print("\nFirst few rows of the binarized DataFrame:")
    print(df_binarized.head())

if __name__ == "__main__":
    main()