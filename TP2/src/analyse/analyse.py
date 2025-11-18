import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_density(df, feature, title=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)
    sns.kdeplot(df[feature], shade=True)
    if title is None:
        title = f'Density of {feature}'
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.show()

def plot_feature_comparison(df, feature1, feature2, kind='box', title=None, figsize=(8, 5)):
    plt.figure(figsize=figsize)
    if kind == 'box':
        sns.boxplot(x=feature2, y=feature1, data=df)
    elif kind == 'scatter':
        sns.scatterplot(x=feature2, y=feature1, data=df)
    else:
        raise ValueError("kind must be 'box' or 'scatter'")
    if title is None:
        title = f'Comparison of {feature1} by {feature2}'
    plt.title(title)
    plt.show()

def profile(dataset):
    pass

def main():
    X_df = pd.read_csv('alt_acsincome_ca_features_85.csv')
    y_df = pd.read_csv('alt_acsincome_ca_labels_85.csv')
    
    # split the dataset into 60% of train data and the rest for validation and test
    X_train, X_new, y_train, y_new = train_test_split(X_df, y_df, test_size=0.4, random_state=42)
    # further split the rest in have for validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_new, y_new, test_size=0.5, random_state=42)
    # In total, there is 60% of train, 20% of val, and 20% of test

    #clf = DecisionTreeClassifier(random_state=42)
    #clf.fit(X_train, y_train)

    #print(X_df.describe())
    plot_feature_comparison(X_df, 'AGEP', 'MAR')

if __name__ == "__main__":
    main()