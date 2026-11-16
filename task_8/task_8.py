import os
import numpy as np
import pandas as pd

from skimage.feature import hog
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from skimage.filters import sobel_h, sobel_v

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cats_dir = 'cats'
nocats_dir = 'no_cats'

X = []
y = []

for filename in os.listdir(cats_dir):
    path = os.path.join(cats_dir, filename)
    image = imread(path)

    X.append(image)
    y.append(1)

for filename in os.listdir(nocats_dir):
    path = os.path.join(nocats_dir, filename)
    image = imread(path)

    X.append(image)
    y.append(0)

y = np.array(y)

print('Всего изображений:', len(X))
print('Кошек:', sum(y == 1))
print('Остальное:', sum(y == 0))

X_resized = []

for image in X:
    resized = resize(image, (100, 100), anti_aliasing=True)
    X_resized.append(resized)

X_resized = np.array(X_resized)

X_hog = []
y_hog = []

for image, label in zip(X_resized, y):

    gray = np.dot(image[..., :3], [0.2125, 0.7154, 0.0721])

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2',
        feature_vector=True
    )

    X_hog.append(features)
    y_hog.append(label)

X_hog = np.array(X_hog)
y_hog = np.array(y_hog)

indices = np.arange(len(y_hog))

train_idx, test_idx, y_train, y_test = train_test_split(
    indices,
    y_hog,
    test_size=0.2,
    random_state=42,
    stratify=y_hog
)

X_train = X_hog[train_idx]
X_test = X_hog[test_idx]

print('Обучающая выборка:', X_train.shape)
print('Тестовая выборка:', X_test.shape)

models = {
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=1, gamma='scale')),
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5)),
    ]),
    'Логистическая регрессия': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=2000)),
    ]),
}

predictions = {}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    predictions[name] = predicted
    results.append({
        'Модель': name,
        'Accuracy': accuracy_score(y_test, predicted),
        'F1 (кошка)': f1_score(y_test, predicted, zero_division=0),
    })

comparison = pd.DataFrame(results).sort_values(
    by=['F1 (кошка)', 'Accuracy'], ascending=False
)
print('\nСравнение моделей на тестовой выборке:')
print(comparison.round(3).to_string(index=False))

best_name = comparison.iloc[0]['Модель']
y_pred = predictions[best_name]

print(f'\nМодель для разбора ошибок: {best_name}')
print('Classification report:')
print(classification_report(
    y_test,
    y_pred,
    target_names=['Остальное', 'Кошка']
))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Остальное', 'Кошка'],
    yticklabels=['Остальное', 'Кошка']
)

plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title(f'Матрица ошибок: {best_name}')
plt.show()

errors = np.where(y_test != y_pred)[0]

print('Количество ошибок:', len(errors))
print('Индексы ошибочных изображений в тестовой выборке:', errors)
if len(errors) == 0:
    print('Ошибок на тестовой выборке нет.')
else:
    fig, axes = plt.subplots(
        1, len(errors),
        figsize=(5 * len(errors), 5)
    )

    if len(errors) == 1:
        axes = [axes]

    for ax, error_idx in zip(axes, errors):

        original_idx = test_idx[error_idx]

        true_label = 'Кошка' if y_test[error_idx] == 1 else 'Остальное'
        pred_label = 'Кошка' if y_pred[error_idx] == 1 else 'Остальное'

        ax.imshow(X_resized[original_idx])
        ax.set_title(
            f'Истинный класс: {true_label}\n'
            f'Предсказание: {pred_label}',
            fontsize=11
        )
        ax.axis('off')

    plt.tight_layout()
    plt.show()