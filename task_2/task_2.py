import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

def read_data(data_file: str, separator: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame pandas."""
    df = pd.read_csv(data_file, encoding="utf-8", sep=separator)
    return df

def get_selected_countries_df(df: pd.DataFrame, selected_countries: list[str]):
    """Фильтрует данные, оставляя только указанные страны."""
    return df[df["Entity"].isin(selected_countries)]

def get_general_data(df: pd.DataFrame):
    temp = get_selected_countries_df(df, ['World'])
    temp = temp[temp['Day'] < '2020-12-01']
    temp = temp[temp['Day'] >= '2020-01-28']
    temp = temp.drop(columns=['Code', 'Entity', 'Year', 'Continent'])
    temp["time_idx"] = np.arange(len(temp))
    return temp

def arima(y_train, test_df):
    model = ARIMA(y_train, order=(2, 1, 2))
    result = model.fit()
    
    predicted = result.forecast(steps = len(test_df))
    return predicted

def linear(t_train, y_train, t_test, y_test, y_predicted):
    plt.figure(figsize=(14, 6))
    plt.scatter(t_train, y_train, label='y (исходные данные)')
    plt.scatter(t_test, y_test, label='y (тестовая выборка)', color='red')
    plt.plot(t_test, y_predicted, label='Прогноз y', color='green')
    plt.xlabel('Время')
    plt.ylabel('Значения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
def knn(train_df, y_train, test_df):
    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(train_df[["time_idx"]], y_train)
    return knn.predict(test_df[["time_idx"]])

def ols(x1_train, y_train, x1_test):
    ols_1 = LinearRegression().fit(x1_train, y_train)
    return ols_1.predict(x1_test)

def svr(x1_train, y_train, x1_test):
    svr = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100.0, epsilon=0.01)),
        transformer=StandardScaler()).fit(x1_train, y_train)
    return svr.predict(x1_test)

def ols2(x2_train, y_train, x2_test):
    ols_2 = LinearRegression().fit(x2_train, y_train)
    return ols_2.predict(x2_test)

def svr2(x2_train, y_train, x2_test):
    svr_2 = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100.0, epsilon=0.01)),
        transformer=StandardScaler()
    ).fit(x2_train, y_train)
    return svr_2.predict(x2_test)

def metrics(dict):
    results = []
    valid = dict['valid']
    
    for model_name, predicted in dict.items():
        mae = mean_absolute_error(valid, predicted)
        rmse = root_mean_squared_error(valid, predicted)
        results.append({"Модель": model_name, "MAE": mae, "RMSE": rmse})
    
    for i in results:
        print(i["Модель"] + ", MAE:" + str(i["MAE"]) + ", RMSE:" + str(i["RMSE"]))
    
def main():
    data_file = "DataSet1_4.csv"  # Путь к файлу с данными
    separator = ","  # Разделитель данных в файле
    y_col = 'Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'
    x2_col = 'Total confirmed cases of COVID-19'
    
    data = read_data(data_file, separator)
    data["Day"] = pd.to_datetime(data["Day"])
    general = get_general_data(data)
    
    split_idx = int(len(general) * 0.8)
    train_df = general.iloc[:split_idx].copy()
    test_df = general.iloc[split_idx:].copy()
    y_train = train_df[y_col].values
    t_train = train_df["time_idx"]
    y_test = test_df[y_col].values
    t_test = test_df["time_idx"]
    
    x1_train = train_df[["time_idx"]]
    x1_test = test_df[["time_idx"]]
    
    x2_train = train_df[["time_idx", x2_col]]
    x2_test = test_df[["time_idx", x2_col]]
    
    predicted_arima = arima(y_train, test_df)
    predicted_knn = knn(train_df, y_train, test_df)
    
    predicted_ols = ols(x1_train, y_train, x1_test)
    predicted_svr = svr(x1_train, y_train, x1_test)
    
    predicted_ols2 = ols2(x2_train, y_train, x2_test)
    predicted_svr2 = svr2(x2_train, y_train, x2_test)

    linear(t_train, y_train, t_test, y_test, predicted_arima)
    linear(t_train, y_train, t_test, y_test, predicted_knn)
    
    linear(t_train, y_train, t_test, y_test, predicted_ols)
    linear(t_train, y_train, t_test, y_test, predicted_svr)
    
    linear(t_train, y_train, t_test, y_test, predicted_ols2)
    linear(t_train, y_train, t_test, y_test, predicted_svr2)
    
    dict_metrics = {}
    dict_metrics['valid'] = test_df[y_col].values
    dict_metrics['arima'] = predicted_arima    
    dict_metrics['knn'] = predicted_knn  
    dict_metrics['ols'] = predicted_ols  
    dict_metrics['svr'] = predicted_svr  
    dict_metrics['ols2'] = predicted_ols2  
    dict_metrics['svr2'] = predicted_svr2  
    
    metrics(dict_metrics)
    
if __name__ == '__main__':
    main()