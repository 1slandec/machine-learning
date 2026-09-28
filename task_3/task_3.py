import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

def read_data(data_file: str, separator: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame pandas."""
    df = pd.read_csv(data_file, encoding="utf-8", sep=separator)
    return df

def ols2(x2_train, y_train, x2_test):
    ols_2 = LinearRegression().fit(x2_train, y_train)
    return ols_2.predict(x2_test)

def ols6(x3_train, y_train, x3_test):
    ols_3 = LinearRegression().fit(x3_train, y_train)
    return ols_3.predict(x3_test)

def svr2(x2_train, y_train, x2_test):
    svr_2 = SVR(kernel='rbf', C=100, gamma='scale', epsilon=10).fit(x2_train, y_train)
    return svr_2.predict(x2_test)

def linear(t_train, y_train, t_test, y_test, y_predicted, name):
    plt.figure(figsize=(14, 6))
    plt.scatter(t_train, y_train, label='y (исходные данные)')
    plt.scatter(t_test, y_test, label='y (тестовая выборка)', color='red')
    plt.plot(t_test, y_predicted, label='Прогноз y', color='green')
    plt.xlabel('Время')
    plt.ylabel('Значения')
    plt.title(f"Предсказанные значения методом {name}", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'task_3/data/method_{name}.png', dpi=150, bbox_inches='tight')
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

    if title is None:
        title = "Скрипичная диаграмма: исходные и предсказанные значения"

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

    # Добавим точки поверх скрипок, чтобы было видно реальную выборку
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

    plt.show()
    plt.close(fig)
    
def main():
    path = 'Classification1.csv'
    separator = ','
    
    data = read_data(path, separator)
    data["time_idx"] = np.arange(len(data))
    threshold = 100
    data['class'] = np.where(data['count']>= threshold, 1, 0)
    # data['temp^2'] = data['temp']**2
    # data['humidity^2'] = data['humidity']**2
    # data['windspeed^2'] = data['windspeed']**2
    # data['summer * temp'] = data['summer']*data['temp']
    # data['summer * humidity'] = data['summer']*data['humidity']
    # data['windspeed * humidity'] = data['windspeed']*data['humidity']
    # data['workingday * humidity'] = data['workingday']*data['humidity']
    # data['holiday * temp'] = data['holiday']*data['temp']
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
    # x3_col = 'humidity'
    # xn_cols = ['workingday', 'holiday', 'windspeed', 'temp', 'humidity', 'summer', 'temp^2', 'humidity^2', 'windspeed^2', 'summer * temp', 'summer * humidity', 'windspeed * humidity','workingday * humidity', 'holiday * temp']
    x2_train = train_df[[x1_col, x2_col]]
    x2_test = test_df[[x1_col, x2_col]] 
    # x6_train = train_df[xn_cols]
    # x6_test = test_df[xn_cols]
    
    predicted_ols2 = ols2(x2_train, y_train, x2_test)
    predicted_ols2_class = np.where(predicted_ols2>threshold, 1, 0)
    predicted_svr2 = svr2(x2_train, y_train, x2_test)
    predicted_svr2_class = np.where(predicted_svr2>threshold, 1, 0)
    # predicted_ols6 = ols6(x6_train, y_train, x6_test)

    linear(t_train, train_df['class'].values, t_test, test_df['class'].values, predicted_ols2_class, 'МНК с двумя параметрами')
    linear(t_train, train_df['class'].values, t_test, test_df['class'].values, predicted_svr2_class, 'SVR с двумя параметрами')
    # linear(t_train, y_train, t_test, y_test, predicted_ols6, 'МНК с 6 параметрами')

    
    predictions = {
        "МНК с двумя параметрами": predicted_ols2,
        "SVR с двумя параметрами": predicted_svr2,
        # "МНК с шестью параметрами": predicted_ols6
    }

    violin_actual_vs_predicted(
        y_true=y_test,
        predictions=predictions,
        title="Распределение фактических и прогнозируемых значений count",
    )

    metrics_dict = {
        "valid": y_test,
        "МНК с двумя параметрами": predicted_ols2,
        "SVR с двумя параметрами": predicted_svr2,
        # "МНК с шестью параметрами": predicted_ols6
    }

    metrics_table = metrics(metrics_dict, y_train=y_train)

    print("\nСводная таблица метрик:")
    print(metrics_table)

if __name__ == "__main__":
    main() 