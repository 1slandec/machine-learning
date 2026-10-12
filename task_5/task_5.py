import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def read_data(data_file: str, separator: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame pandas."""
    df = pd.read_csv(data_file, encoding="utf-8", sep=separator)
    return df

def scaler(x_test, x_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def main():
    path = 'Classification1.csv'
    path2 = 'online_retail_II.csv'
    separator = ','
    separator2 = ';'
    data = read_data(path, separator)
    data2 = read_data(path2, separator2)
    data["time_idx"] = np.arange(len(data))
    threshold = 100
    data['class'] = np.where(data['count']>= threshold, 1, 0)
    data = data.iloc[:int(len(data)*0.1)]

    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    y_train_class = train_df['count'].values
    y_test_class = test_df['count'].values

    cols = ['count', 'holiday', 'humidity', 'summer', 'temp', 'windspeed', 'workingday']
    x1_col = 'workingday'
    x2_col = 'temp'
    x_scaled = StandardScaler().fit_transform(data[cols])
    y_class = np.concatenate([y_train_class, y_test_class])
    x2_train = train_df[[x1_col, x2_col]]
    x2_test = test_df[[x1_col, x2_col]] 

    x2_train_scaled, x2_test_scaled = scaler(x2_test, x2_train)

    inertia = []
    silhouette = []
    k_range = range(2, 7)
    for k in k_range:
        k_means = KMeans(n_clusters=k, random_state=42, n_init = 10)
        k_means.fit(x_scaled)
        inertia.append(k_means.inertia_)
        silhouette.append(silhouette_score(x_scaled, k_means.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_range, inertia, marker='o')
    axes[0].set_title('Метод локтя')
    axes[0].set_xlabel('Число кластеров k')
    axes[0].set_ylabel('Инерция (WCSS)')

    # График силуэта
    axes[1].plot(k_range, silhouette, marker='o', color='orange')
    axes[1].set_title('Метод силуэта')
    axes[1].set_xlabel('Число кластеров k')
    axes[1].set_ylabel('Число силуэта')

    plt.show()


    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_scaled)

    k_range = range(2, 8)
    models = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x_scaled)
        models[k] = km

    plt.figure(figsize=(15, 10))

    for i, k in enumerate(k_range):
        kmeans_model = models[k]
        labels = kmeans_model.labels_
        centers = kmeans_model.cluster_centers_

        centers_pca = pca.transform(centers)

        plt.subplot(2, 3, i + 1)
        plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=30, alpha=0.7
        )   
        plt.scatter(
            centers_pca[:, 0],
            centers_pca[:, 1],
            c='red',
            marker='x',
            s=100,
            linewidths=3,
            label='Центры',
        )

        plt.title(f'k = {k}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()