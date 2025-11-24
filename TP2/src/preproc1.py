import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(feature_path, label_path):
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path, header=None, names=["label"])

    print("Valeurs uniques de labels (avant nettoyage) :")
    print(labels["label"].unique())

    # Supprimer les lignes de type 'PINCP'
    labels = labels[labels["label"] != "PINCP"].reset_index(drop=True)

    # Synchroniser tailles
    min_len = min(len(features), len(labels))
    features = features.iloc[:min_len].reset_index(drop=True)
    labels = labels.iloc[:min_len].reset_index(drop=True)

    # Colonnes sélectionnées
    categorical_columns = ['COW', 'MAR', 'OCCP', 'POBP', 'RELP','RAC1P']
    numeric_columns = ['AGEP', 'WKHP','SCHL']

    X = features[categorical_columns + numeric_columns]

    # Nettoyage du label
    y = labels["label"].astype(str).str.strip()

    # Convertir True/False en 1/0
    y = y.replace({"True": 1, "False": 0})

    # Reconvertir en entier
    y = y.astype(int)

    # Encodage OneHot
    column_transformer = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns),
        ],
        remainder='passthrough'
    )

    X_encoded = column_transformer.fit_transform(X)

    # Récupérer noms colonnes
    encoded_feature_names = column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_columns)
    final_feature_names = list(encoded_feature_names) + numeric_columns

    X_preprocessed = pd.DataFrame(X_encoded, columns=final_feature_names)

    # Vérification finale
    print("Aligned shapes → X:", X_preprocessed.shape, ", y:", y.shape)

    return X_preprocessed, y
