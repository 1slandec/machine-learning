import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def read_data(data_file: str, separator: str, dec: str) -> pd.DataFrame:
    df = pd.read_csv(data_file, encoding="utf-8", sep=separator, decimal=dec)
    return df


def prepare_data(data):
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%d.%m.%Y %H:%M')
    data = data.dropna(subset=['Customer ID'])
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
    data = data[data['Price'] > 0]
    data = data[data['Quantity'] > 0]
    data = data[~data['Invoice'].astype(str).str.startswith('C')]
    return data


def profiles(x_scaled, cols, model, num):
    profile = pd.DataFrame(x_scaled, columns=cols)
    profile['cluster'] = model.labels_
    means = profile.groupby('cluster').mean()

    means.T.plot(kind='bar', figsize=(10, 5))
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title(f'Средние значения признаков по кластерам (k = {num})')
    plt.ylabel('Стандартизованное значение')
    plt.savefig(f'task_5/data/cluster_profiles{num}.png', dpi=150, bbox_inches='tight')
    plt.show()


def sizes(model, num):
    pd.Series(model.labels_).value_counts().sort_index().plot(kind='bar')
    plt.title(f'Размер кластеров (k = {num})')
    plt.xlabel('Кластер')
    plt.ylabel('Число наблюдений')
    plt.savefig(f'task_5/data/cluster_sizes{num}.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.makedirs('task_5/data', exist_ok=True)

    path = 'Classification1.csv'
    path2 = 'online_retail_II.csv'
    data = read_data(path, ',', '.')
    data2 = read_data(path2, ';', ',')
    data2 = prepare_data(data2)

    cols = ['holiday', 'humidity', 'summer', 'temp', 'windspeed', 'workingday']
    x_scaled = StandardScaler().fit_transform(data[cols])

    inertia = []
    silhouette = []
    k_range = range(2, 7)
    for k in k_range:
        k_means = KMeans(n_clusters=k, random_state=42, n_init=10)
        k_means.fit(x_scaled)
        inertia.append(k_means.inertia_)
        silhouette.append(silhouette_score(x_scaled, k_means.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(k_range, inertia, marker='o')
    axes[0].set_title('Метод локтя')
    axes[0].set_xlabel('Число кластеров k')
    axes[0].set_ylabel('Инерция (WCSS)')

    axes[1].plot(k_range, silhouette, marker='o', color='orange')
    axes[1].set_title('Метод силуэта')
    axes[1].set_xlabel('Число кластеров k')
    axes[1].set_ylabel('Коэффициент силуэта')

    plt.savefig('task_5/data/Silhouette_Elbow.png', dpi=150, bbox_inches='tight')
    plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_scaled)

    k_range = range(2, 8)
    models = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x_scaled)
        models[k] = km

    for k in k_range:
        profiles(x_scaled, cols, models[k], k)
        sizes(models[k], k)

    plt.figure(figsize=(15, 10))
    for i, k in enumerate(k_range):
        centers_pca = pca.transform(models[k].cluster_centers_)

        plt.subplot(2, 3, i + 1)
        plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=models[k].labels_, cmap='tab10', s=30, alpha=0.7
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
    plt.savefig('task_5/data/Clusters.png', dpi=150, bbox_inches='tight')
    plt.show()

    data_for_apriori = (
        data2.groupby(['Invoice', 'Description'])['Price']
        .sum()
        .unstack(fill_value=0)
        .fillna(0)
        .map(lambda x: 1 if x > 0 else 0)
    )

    frequent_itemsets = apriori(data_for_apriori.astype('bool'), min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.02)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    rule_sets = {
        'support': rules[rules['support'] >= 0.025],
        'lift': rules[rules['lift'] >= 13.6],
        'confidence': rules[rules['confidence'] >= 0.6],
    }

    for name, item in rule_sets.items():
        plt.figure(figsize=(10, 6))

        scatter = plt.scatter(
            item['support'],
            item['confidence'],
            c=item['lift'],
            cmap='viridis',
            s=60,
            alpha=0.8,
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label('Lift (Сила связи)')

        plt.xlabel('Support (Поддержка)', fontsize=12)
        plt.ylabel('Confidence (Достоверность)', fontsize=12)
        plt.title(
            'Ассоциативные правила: Support vs Confidence (Цвет = Lift)', fontsize=14
        )
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.savefig(f'task_5/data/apriori_{name}.png', dpi=150, bbox_inches='tight')
        plt.show()

        print(f'top-5 po {name}')
        print(item.sort_values(name, ascending=False).head(5).to_string())


if __name__ == "__main__":
    main()