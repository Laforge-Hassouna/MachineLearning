import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize_features(X, features, method="standard"):
    """
    Normalize numerical features in X.
    method: "standard" -> StandardScaler (mean=0, std=1)
            "minmax"   -> MinMaxScaler (0 to 1)
    """

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unknown normalization method. Use 'standard' or 'minmax'.")

    # Fit + transform numerical features
    X[features] = scaler.fit_transform(X[features])

    return X


def binarize_features(df, features_to_bin: list[str]):
    """
    Binarize specified categorical features in a DataFrame using OneHotEncoder.

    Parameters:
    - df: pandas DataFrame
    - features_to_bin: list of column names to binarize

    Returns:
    - DataFrame with original columns dropped and new one-hot encoded columns added
    """
    # Initialize the encoder
    encoder = OneHotEncoder(sparse_output=False, drop='if_binary')

    # Fit and transform the specified columns
    encoded_data = encoder.fit_transform(df[features_to_bin])

    # Get feature names for the encoded columns
    feature_names = encoder.get_feature_names_out(features_to_bin)

    # Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

    # Drop the original columns and concatenate the encoded ones
    df = df.drop(columns=features_to_bin)
    df = pd.concat([df, encoded_df], axis=1)

    return df
