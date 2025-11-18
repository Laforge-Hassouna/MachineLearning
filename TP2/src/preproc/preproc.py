import pandas as pd
import sys
import os

import preproclib as ppl

def read_features_and_labels(features_path, labels_path):
    """
    Read features and labels from separate CSV files.
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    return X, y

def preprocess_data(X, y):
    """
    Apply preprocessing steps to features and labels.

    Replace the body of this function with your real preprocessing logic.
    """
    acsincome_features_to_bin = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P']
    X = ppl.binarize_features(X, acsincome_features_to_bin)

    return X, y


def save_processed_data(X, y, output_dir, features_path, labels_path):
    """
    Save the processed features and labels to CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Remove extension from filenames
    features_filename = os.path.splitext(os.path.basename(features_path))[0]
    labels_filename = os.path.splitext(os.path.basename(labels_path))[0]

    # Save processed files
    X.to_csv(f"{output_dir}/{features_filename}_processed.csv", index=False)
    y.to_csv(f"{output_dir}/{labels_filename}_processed.csv", index=False)

    print(f"Processed data saved to {output_dir}/")


def main():
    if len(sys.argv) < 4:
        print("Usage: python preprocess_and_save.py <output_dir> <features.csv> <labels.csv>")
        sys.exit(1)

    output_dir = sys.argv[1]
    features_path = sys.argv[2]
    labels_path = sys.argv[3]

    X, y = read_features_and_labels(features_path, labels_path)
    X_proc, y_proc = preprocess_data(X, y)
    save_processed_data(X_proc, y_proc, output_dir, features_path, labels_path)


if __name__ == "__main__":
    main()