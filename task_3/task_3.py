import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


def read_data(data_file: str, separator: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame pandas."""
    df = pd.read_csv(data_file, encoding="utf-8", sep=separator)
    return df

def ols(x_train, y_train, x_test):
    ols = LinearRegression().fit(x_train, y_train)
    return ols.predict(x_test)

def svr(x_train, y_train, x_test, e):
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=e).fit(x_train, y_train)
    return svr.predict(x_test)

def svm(x_train, y_train, x_test):
    param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 0.1, 1]
}
    _svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.1, probability=True, gamma='scale')).fit(x_train, y_train)
    grid = GridSearchCV(_svm, param_grid, cv=5, scoring='accuracy')
    grid.fit(x_train, y_train)
    # print(grid.best_params_)
    # print(grid.best_score_)
    return (_svm.predict(x_test)>=0.5).astype(int), _svm.predict_proba(x_test)



def knn(x_train, y_train, x_test):
    _knn = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)
    return (_knn.predict(x_test)>=0.5).astype(int), _knn.predict_proba(x_test)

def linear(t_train, y_train, t_test, y_test, y_predicted, name):
    plt.figure(figsize=(14, 6))
    # plt.scatter(t_train, y_train, label='y (исходные данные)')
    plt.scatter(t_test, y_test, label='y (тестовая выборка)', color='red')
    plt.plot(t_test, y_predicted, label='Прогноз y', color='green')
    plt.xlabel('Время')
    plt.ylabel('Значения')
    plt.title(f"Предсказанные значения методом {name}", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'task_3/data/linear_method_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
def _mase(y_true, y_pred, y_scale):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_scale = np.asarray(y_scale, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую длину.")
    numerator = mean_absolute_error(y_true, y_pred)
    if y_scale.size < 2:
        return np.nan
    denominator = np.mean(np.abs(np.diff(y_scale)))
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0 if numerator == 0 else np.nan
    return numerator / denominator
    
def metrics(metrics_dict, y_train=None):

    results = []
    valid = np.asarray(metrics_dict["valid"], dtype=float).ravel()
    if y_train is None:
        y_train = metrics_dict.get("train", valid)
    y_train = np.asarray(y_train, dtype=float).ravel()
    reserved_keys = {"valid", "train"}
    for model_name, predicted in metrics_dict.items():
        if model_name in reserved_keys:
            continue
        predicted = np.asarray(predicted, dtype=float).ravel()
        mae = mean_absolute_error(valid, predicted)
        rmse = root_mean_squared_error(valid, predicted)
        mape = mean_absolute_percentage_error(valid, predicted) * 100.0
        mase = _mase(
            y_true=valid,
            y_pred=predicted,
            y_scale=y_train
        )
        results.append({
            "Модель": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "MASE": mase,
        })

    return pd.DataFrame(results)

def violin_actual_vs_predicted(
    y_true,
    predictions: dict,
    title: str = None,
    figsize=(10, 6),
):
    y_true = np.asarray(y_true).ravel()
    datasets = [y_true]
    labels = ["Исходные данные"]
    for model_name, y_pred in predictions.items():
        y_pred = np.asarray(y_pred).ravel()
        if y_pred.size != y_true.size:
            raise ValueError(
                f"Длина предсказания '{model_name}' не совпадает "
                f"с длиной исходных данных."
            )
        datasets.append(y_pred)
        labels.append(str(model_name))
    positions = np.arange(1, len(datasets) + 1)
    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(
        datasets,
        positions=positions,
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % len(colors)])
        body.set_edgecolor("black")
        body.set_alpha(0.65)
    rng = np.random.default_rng(42)
    for i, data in enumerate(datasets, start=1):
        jitter = rng.normal(0, 0.035, size=len(data))
        ax.scatter(
            i + jitter,
            data,
            s=12,
            alpha=0.45,
            color="black",
            linewidths=0,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_xlabel("Тип данных")
    ax.set_ylabel("Значения")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.savefig(f'task_3/data/violin_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
def class_metrics(metrics_dict):
    results = []
    valid = np.asarray(metrics_dict["valid"], dtype=float).ravel()
    reserved_keys = {"valid", "train"}
    for model_name, predicted in metrics_dict.items():
        if model_name in reserved_keys:
            continue
        predicted = np.asarray(predicted, dtype=float).ravel()
        accuracy = accuracy_score(valid, predicted)
        f_score = f1_score(valid, predicted)
        results.append({
                "Модель": model_name,
                "accuracy": accuracy,
                "f_score": f_score,
            })
        _confusion_matrix(predicted, valid, model_name)
    
    return pd.DataFrame(results)

def _confusion_matrix(y_pred_class, y_class, name):
    
    conf_matrix = confusion_matrix(y_pred=y_pred_class, y_true=y_class)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix для {name}')
    plt.savefig(f'task_3/data/conf_method_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
def roc(y_class, y_predict_proba, name):
    fpr, tpr, thresholds = roc_curve(y_true=y_class, y_score=y_predict_proba[:, 1])

    # Рассчитать площадь под ROC-кривой (AUC-ROC)
    roc_auc = auc(fpr, tpr)

    # Визуализировать ROC-кривую
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    # plt.plot(fpr, color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='случайный классификатор')  # случайный классификатор
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC-кривая для {name}')
    plt.legend(loc='lower right')
    plt.savefig(f'task_3/data/roc_{name}.png', dpi=150, bbox_inches='tight')

    plt.show()


def main():
    path = 'Classification1.csv'
    separator = ','
    
    data = read_data(path, separator)
    data["time_idx"] = np.arange(len(data))
    threshold = 100
    data['class'] = np.where(data['count']>= threshold, 1, 0)
    data = data.iloc[:int(len(data)*0.1)]
    y_col = 'count'

    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()
    y_train = train_df[y_col].values
    t_train = train_df["time_idx"]
    y_test = test_df[y_col].values
    t_test = test_df["time_idx"]
    
    x1_col = 'workingday'
    x2_col = 'temp'
    x_train = train_df[[x1_col]]
    x_test = test_df[[x1_col]]
    xn_cols = ['workingday', 'holiday', 'windspeed', 'temp', 'humidity', 'summer']
    x2_train = train_df[[x1_col, x2_col]]
    x2_test = test_df[[x1_col, x2_col]] 
    x6_train = train_df[xn_cols]
    x6_test = test_df[xn_cols]
    
    predicted_ols2 = ols(x2_train, y_train, x2_test)
    predicted_ols2_class = np.where(predicted_ols2>threshold, 1, 0)
    predicted_svr2 = svr(x2_train, y_train, x2_test, 10)
    predicted_svr2_class = np.where(predicted_svr2>threshold, 1, 0)
    class_svm, svm_proba = svm(x2_train, train_df['class'].values, x2_test)
    class_knn, knn_proba = knn(x2_train, train_df['class'].values, x2_test)

    # linear(t_train, y_train, t_test, y_test, predicted_ols2, 'МНК с двумя параметрами')
    # linear(t_train, y_train, t_test, y_test, predicted_svr2, 'SVR с двумя параметрами')

    # linear(t_train, train_df['class'].values, t_test, test_df['class'].values, predicted_ols2_class, 'МНК с двумя параметрами')
    # linear(t_train, train_df['class'].values, t_test, test_df['class'].values, predicted_svr2_class, 'SVR с двумя параметрами')
    
    linear(t_train, train_df['class'].values, t_test, test_df['class'].values, class_svm, 'SVM')
    linear(t_train, train_df['class'].values, t_test, test_df['class'].values, class_knn, 'КНН')


    
    predictions = {
        "МНК с двумя параметрами": predicted_ols2,
        "SVR с двумя параметрами": predicted_svr2,
    }

    # violin_actual_vs_predicted(
    #     y_true=y_test,
    #     predictions=predictions,
    #     title="Распределение фактических и прогнозируемых значений count",
    # )

    metrics_dict = {
        "valid": y_test,
        "МНК с двумя параметрами": predicted_ols2,
        "SVR с двумя параметрами": predicted_svr2,
    }

    metrics_table = metrics(metrics_dict, y_train=y_train)

    print("\nСводная таблица метрик:")
    print(metrics_table)
    
    class_metrics_dict = {
        "valid": test_df['class'].values,
        "МНК с двумя параметрами": predicted_ols2_class,
        "SVR с двумя параметрами": predicted_svr2_class,
        "КНН": class_knn,
        "SVM": class_svm
    }
    
    metrics_class_table = class_metrics(class_metrics_dict)
    print("\nСводная таблица метрик классификации:")
    print(metrics_class_table)
    
    roc(test_df['class'].values, svm_proba,'SVM')
    roc(test_df['class'].values, knn_proba, "KNN")
    print(f'Log_loss for SVM: {log_loss(test_df['class'], svm_proba)}')
    print(f'Log_loss for KNN: {log_loss(test_df['class'], knn_proba)}')
if __name__ == "__main__":
    main() 