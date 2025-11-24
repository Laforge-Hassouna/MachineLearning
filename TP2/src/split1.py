import pandas as pd
from sklearn.model_selection import train_test_split
from preproc1 import preprocess_data


def split_data():
    X, y = preprocess_data("../Dataset/alt_acsincome_ca_features_85.csv", "../Dataset/alt_acsincome_ca_labels_85.csv")


    # Split en 60% train, 20% val, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()
    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)