import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split

def read_features_and_labels(features_path, labels_path):
    """
    Read features and labels from separate CSV files.
    """
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and labels into train/test and save as CSV files.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    return X_train, X_test, y_train, y_test

def save_data(X_train, X_test, y_train, y_test, output_dir, features_path, labels_path):

    os.makedirs(output_dir, exist_ok=True)

    features_filename = os.path.splitext(os.path.basename(features_path))[0]
    labels_filename = os.path.splitext(os.path.basename(labels_path))[0]

    # Save features
    X_train.to_csv(f'{output_dir}/{features_filename}_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/{features_filename}_test.csv', index=False)

    # Save labels (as Series, so flatten to 1D array for consistency)
    y_train.to_csv(f'{output_dir}/{labels_filename}_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/{labels_filename}_test.csv', index=False)

def main():
    if len(sys.argv) < 4:
        print("Usage: python split_and_save.py <output_dir> <features.csv> <labels.csv>")
        sys.exit(1)

    output_dir = sys.argv[1]
    features_path = sys.argv[2]
    labels_path = sys.argv[3]

    X, y = read_features_and_labels(features_path, labels_path)
    split_result = split_data(X, y)
    save_args = (*split_result, output_dir, features_path, labels_path)
    save_data(*save_args)

    print(f"Train/test sets saved to {output_dir}/")

if __name__ == "__main__":
    main()
