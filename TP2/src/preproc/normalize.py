import pandas as pd
import sys
import os

import preproclib as ppl


def read_features(features_path):
    """
    Read features and labels from separate CSV files.
    """
    X = pd.read_csv(features_path)
    return X


def normalize_data(X, method):
    """
    Apply preprocessing steps to features and labels.
    """
    features = ['AGEP', 'WKHP']
    X = ppl.normalize_features(X, features, method=method)

    return X


def save_normalized_data(X, output_dir, features_path, suffix="normalized"):
    """
    Save normalized features and labels to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    features_filename = os.path.splitext(os.path.basename(features_path))[0]

    X.to_csv(f"{output_dir}/{features_filename}_{suffix}.csv", index=False)

    print(f"Normalized data saved to {output_dir}/")


def main():
    if len(sys.argv) < 4:
        print("Usage: python normalize_and_save.py <output_dir> <features.csv> <standard|minmax>")
        sys.exit(1)

    output_dir = sys.argv[1]
    features_path = sys.argv[2]
    
    method = sys.argv[3]

    X = read_features(features_path)
    X_norm = normalize_data(X, method=method)
    save_normalized_data(X_norm, output_dir, features_path)


if __name__ == "__main__":
    main()
