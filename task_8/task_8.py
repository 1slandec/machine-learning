import os
from pathlib import Path

import numpy as np
import pandas as pd

from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cats_dir = 'cats'
nocats_dir = 'no_cats'
data_dir = Path('task_8/data')
class_names = ['Остальное', 'Кошка']

X = []
y = []
filenames = []

for filename in os.listdir(cats_dir):
    path = os.path.join(cats_dir, filename)
    image = imread(path)

    X.append(image)
    y.append(1)
    filenames.append(filename)

for filename in os.listdir(nocats_dir):
    path = os.path.join(nocats_dir, filename)
    image = imread(path)

    X.append(image)
    y.append(0)
    filenames.append(filename)

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

    # Переводим изображение в оттенки серого
    gray = np.dot(image[..., :3], [0.2125, 0.7154, 0.0721])

    # Извлекаем HOG-признаки
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

# Для визуального разбора берём модель с наибольшим F1 на этом тесте.
best_name = comparison.iloc[0]['Модель']
y_pred = predictions[best_name]

data_dir.mkdir(exist_ok=True)

# Одна строка на исходное изображение с именем, меткой и HOG-признаками.
hog_table = pd.DataFrame(
    X_hog, columns=[f'hog_{i}' for i in range(X_hog.shape[1])]
)
hog_table.insert(0, 'label', y_hog)
hog_table.insert(0, 'filename', filenames)
hog_table.to_csv(data_dir / 'hog_features.csv', index=False, encoding='utf-8-sig')

comparison.to_csv(data_dir / 'model_comparison.csv', index=False, encoding='utf-8-sig')

test_predictions = pd.DataFrame({
    'filename': [filenames[i] for i in test_idx],
    'true_label': y_test,
    **predictions,
})
test_predictions.to_csv(
    data_dir / 'test_predictions.csv', index=False, encoding='utf-8-sig'
)

print(f'\nМодель для разбора ошибок: {best_name}')
print('Classification report:')
print(classification_report(
    y_test,
    y_pred,
    target_names=class_names,
    zero_division=0,
))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
)

ax.set_xlabel('Предсказанный класс')
ax.set_ylabel('Истинный класс')
ax.set_title(f'Матрица ошибок: {best_name}')
fig.tight_layout()
fig.savefig(data_dir / 'confusion_matrix.png', dpi=150)
plt.show()
plt.close(fig)


def save_examples(example_indices, title, output_path):
    count = len(example_indices)
    columns = min(5, max(1, count))
    rows = max(1, (count + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis('off')

    for ax, test_index in zip(axes, example_indices):
        original_index = test_idx[test_index]
        true_label = class_names[y_test[test_index]]
        predicted_label = class_names[y_pred[test_index]]
        ax.imshow(X_resized[original_index])
        ax.set_title(f'Истинный класс: {true_label}\nПредсказание: {predicted_label}')

    if count == 0:
        axes[0].text(
            0.5, 0.5, 'Таких примеров нет',
            ha='center', va='center', transform=axes[0].transAxes,
        )

    fig.suptitle(f'{title} — {best_name}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    if count:
        plt.show()
    plt.close(fig)


correct = y_test == y_pred
correct_examples = np.concatenate([
    np.flatnonzero(correct & (y_test == label))[:3] for label in (1, 0)
])
errors = np.flatnonzero(~correct)

print('Количество ошибок:', len(errors))
print('Индексы ошибочных изображений в тестовой выборке:', errors)

save_examples(
    correct_examples, 'Правильные предсказания', data_dir / 'correct_predictions.png'
)
save_examples(errors, 'Ошибочные предсказания', data_dir / 'incorrect_predictions.png')
