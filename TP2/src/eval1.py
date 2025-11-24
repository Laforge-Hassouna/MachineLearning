
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from train_model1 import train_models


def evaluate():
    models, X_test, y_test = train_models()

    accuracies = {}

    for name, model in models.items():
        print(f"\n====== {name} ======")
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {acc}")
        print("Confusion matrix:\n", cm)
        print("Classification report:\n", report)

        # Afficher la matrice de confusion
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matrice de confusion - {name}")
        plt.xlabel("Prédit")
        plt.ylabel("Réel")
        plt.tight_layout()
        plt.savefig(f"{name}_confusion_matrix.png")
        plt.close()

        accuracies[name] = acc

    # Comparaison des accuracy
    plt.figure(figsize=(6,5))
    plt.bar(accuracies.keys(), accuracies.values())
    plt.ylim(0.5, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Comparaison des modèles – Accuracy")
    plt.savefig("comparaison_accuracy.png")
    plt.close()

if __name__ == "__main__":
    evaluate()
