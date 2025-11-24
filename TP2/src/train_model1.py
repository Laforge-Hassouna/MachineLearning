# train_model.py

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from split1 import split_data


def train_models():
    X_train, X_val, X_test, y_train, y_val, y_test = split_data()
    models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }


    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained.")


    return trained_models, X_test, y_test


if __name__ == "__main__":
    train_models()
