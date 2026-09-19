import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import lagrange

def read_data(data_file: str, separator: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame pandas."""
    df = pd.read_csv(data_file, encoding="utf-8", sep=separator)
    return df

def get_selected_countries_df(df: pd.DataFrame, selected_countries: list[str]):
    """Фильтрует данные, оставляя только указанные страны."""
    return df[df["Entity"].isin(selected_countries)]

def get_linear_chart_data(df: pd.DataFrame):
    """Фильтрует данные, оставляя только общемировые показатели заболеваемости еженедельно"""
    result = get_selected_countries_df(df, ["World"])
    result = result.drop(
        columns=[
            "Entity",
            "Code",
            "Total confirmed cases of COVID-19",
            "Year",
            "Continent",
        ]
    )
    return result

def get_linear_graph(df: pd.DataFrame):
    """Строит линейный график (диаграмму)"""

    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(14, 7))
    df.plot(
        ax=ax,
        x="Day",
        y="Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
        marker="o",
        markersize=4,
        linewidth=2,
    )

    ax.set_title(
        f"Динамика изменения количества заболевших COVID-19 во всем мире с 22-01-2020 по 24-04-2022",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Год", fontsize=12, fontweight="bold")
    ax.set_ylabel("Новые подтвержденные случаи болезни", fontsize=12, fontweight="bold")
    ax.get_legend().remove()

    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))
    ax.grid(True, color="gray")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, '^k', transform=ax.transAxes, clip_on=False)

    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.tight_layout()
    plt.savefig(f'task_1/data/all_time_period_world_linear_chart.png', dpi=150, bbox_inches='tight')
    plt.show()

def get_pie_chart_data(df: pd.DataFrame):
    """Суммирует количество заболевших за все время пандемии по континентам"""
    only_continents = get_selected_countries_df(
        df,
        [
            "Africa",
            "Asia",
            "Europe",
            "North America",
            "South America",
            "Oceania",
            "Australia",
        ],
    )
    max_per_country = only_continents.groupby(["Entity"])[
        "Total confirmed cases of COVID-19"
    ].max()
    return max_per_country

def get_pie_chart(df: pd.DataFrame):
    """Строит круговую диаграмму"""
    plt.figure(figsize=(8, 8))
    plt.pie(df, labels=df.index, autopct="%1.1f%%", startangle=140)
    plt.title(f"Доля заболевших на каждом континенте за все время", fontweight="bold")
    plt.legend(title="Континенты", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.tight_layout()
    plt.savefig(f'task_1/data/all_time_continents_pie_chart.png', dpi=150, bbox_inches='tight')
    plt.show()

def get_bar_chart_data(df: pd.DataFrame):
    """Оставляет только 15 стран с наибольшим числом заболевших на конец пандемии"""
    entities_to_delete = [
        "High income",
        "Low income",
        "Lower middle income",
        "Upper middle income",
        "World",
        "World excl. China",
        "World excl. China and South Korea",
        "World excl. China, South Korea, Japan and Singapore",
        "Asia excl. China",
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "South America",
        "Oceania",
        "European Union"
    ]

    max_dates = df.groupby("Entity")["Day"].max()
    temp = df[~df["Entity"].isin(entities_to_delete)]
    temp["date_is_last"] = temp.apply(
        lambda row: row["Day"] == max_dates[row["Entity"]], axis=1
    )
    result = temp[temp["date_is_last"]]
    result = result.drop(
        columns=[
            "Code",
            "Day",
            "Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
            "Year",
            "date_is_last",
            "Continent",
        ]
    )

    return result.nlargest(15, "Total confirmed cases of COVID-19")

def get_bar_chart(df: pd.DataFrame):
    """Строит столбчатую диаграмму"""
    _, ax = plt.subplots(figsize=(15, 9))
    bars = ax.bar(df["Entity"], df["Total confirmed cases of COVID-19"])

    ax.set_title(f"Топ-15 стран по случаям COVID-19", fontweight="bold", pad=20)
    ax.set_xlabel("Регион", fontweight="bold")
    ax.set_ylabel("Количество заболевших", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.0f"))

    plt.xticks(rotation=45, ha="right")
    plt.bar_label(bars, padding=3, fmt="%.0f")
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.tight_layout()
    plt.savefig(f'task_1/data/top_15_countries_all_time_bar_chart.png', dpi=150, bbox_inches='tight')
    plt.show()

def get_violin_plot_data(df: pd.DataFrame):
    """Приводит данные к логарифмической шкале"""
    df = df[~df['Continent'].isin(['Antarctica'])]
    df["log_daily"] = np.log10(df["Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)"] + 1)
    df = df.drop(columns=['Code', 'Year'])
    return df

def get_violin_plot(df: pd.DataFrame):
    """Строит скрипичную диаграмму по ежедневным данным по континентам"""
    _, ax = plt.subplots()
    sns.violinplot(data=df, x="Continent", y="log_daily", inner="quartile")
    plt.xlabel('Континент')
    plt.ylabel("Количество ежедневных случаев болезни")
    plt.title("Распределение заболеваемости по континентам (логарифмическая шкала)")
    ax.set_yticklabels([0,1,10,100,1000,10000,100000,1000000])    

    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.savefig(f'task_1/data/daily_data_continents_violin_plot.png', dpi=150, bbox_inches='tight')
    plt.show()

def get_bad_data(df: pd.DataFrame):
    """Оставляет значения только во всем мире за ~первый месяц, только каждое второе значение заболевших в день"""
    df = get_selected_countries_df(df, ['World'])
    df = df[df['Day'] < '2020-03-01']
    df = df[df['Day'] >= '2020-01-28']
    df = df.drop(columns= ['Code', 'Total confirmed cases of COVID-19', 'Year', 'Continent'])
    df = df.reset_index(drop=True)
    df.loc[df.index[1::2], 'Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'] = np.nan
    return df

def get_valid_data(df: pd.DataFrame):
    """Оставляет значения только во всем мире за ~первый месяц"""
    df = get_selected_countries_df(df, ['World'])
    df = df[df['Day'] < '2020-03-01']
    df = df[df['Day'] >= '2020-01-28']
    df = df.drop(columns= ['Code', 'Total confirmed cases of COVID-19', 'Year', 'Continent'])
    df = df.reset_index(drop=True)
    return df

def fill_average(df: pd.DataFrame):
    """Заполняет пропуски средним значением ряда"""
    average = df['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'].mean()
    df["Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)"]= df["Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)"].fillna(average)
    return df

def fill_median(df: pd.DataFrame):
    """Заполняет пропуски медианным значением ряда"""
    median = df['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'].median()
    df["Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)"]= df["Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)"].fillna(median)
    return df

def fill_average_of_3(df: pd.DataFrame):
    """Заполняет пропуски средним из трех предыдущих элементов"""
    for i in range(1, len(df), 2):
        if i>2:
            prev_1 = df.loc[i-1, 'Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)']
            prev_2 = df.loc[i-2, 'Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)']
            prev_3 = df.loc[i-3, 'Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)']
            value = (prev_1+prev_2+prev_3)/3
        else:
            value = df.at[i-1,'Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)']
        df.at[i, 'Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'] = value
    return df

def fill_lagrange(df: pd.DataFrame):
    """Заполняет пропуски интерполяционным методом Лагранжа"""
    series = df['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)']
    valid_data = series.dropna()
    x_valid = valid_data.index.to_numpy()
    y_valid = valid_data.to_numpy()
    
    polynom = lagrange(x_valid, y_valid)
    
    x_missing = series[series.isna()].index.to_numpy()
    
    filled_series = series.copy()
    filled_series.loc[x_missing] = polynom(x_missing)
    
    df['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'] = filled_series
    return df

def fill_cube_spline(df: pd.DataFrame):
    """Заполняет пропуски методом кубических сплайнов"""
    df['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'] = df['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'].interpolate(method='cubicspline')
    return df

def get_linear_compare(df_valid: pd.DataFrame, df_avg: pd.DataFrame, df_median: pd.DataFrame, df_avg_of_3: pd.DataFrame, df_lagrange: pd.DataFrame, df_cube_spline: pd.DataFrame):
    """Строит линейный график сравнения методов заполнения пропусков"""
    plt.style.use("seaborn-v0_8-whitegrid")
    _, ax = plt.subplots(figsize=(14, 7))
    
    df_avg.plot(
                ax=ax,
                x="Day",
                y="Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
                marker="o",
                markersize=2,
                linewidth=1,
                label='Средние значения ряда',
                color = 'blue',
            )
    
    df_median.plot(
                ax=ax,
                x="Day",
                y="Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
                marker="o",
                markersize=2,
                linewidth=1,
                label='Медианные значения ряда',
                color='red'
            )
    
    df_avg_of_3.plot(
                ax=ax,
                x="Day",
                y="Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
                marker="o",
                markersize=2,
                linewidth=1,
                label='Средние из последних 3-х',
                color = 'yellow'
            )
    
    df_lagrange.plot(
                ax=ax,
                x="Day",
                y="Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
                marker="o",
                markersize=2,
                linewidth=1,
                label='Метод Лагранжа',
                color = 'purple'
            )
    
    df_cube_spline.plot(
                ax=ax,
                x="Day",
                y="Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
                marker="o",
                markersize=2,
                linewidth=1,
                label='Метод кубических сплайнов',
                color = 'black'
            )
    
    df_valid.plot(
                ax=ax,
                x="Day",
                y="Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)",
                marker="o",
                markersize=4,
                linewidth=2,
                label='Правильные значения',
                color = 'green'
            )
    
    ax.set_title(
            f"Сравнение методов заполнения пропусков",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
    ax.set_xlabel("День Марта 2020", fontsize=12, fontweight="bold")
    ax.set_ylabel("Новые подтвержденные случаи болезни", fontsize=12, fontweight="bold")
    y_col = "Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)"
    y_min = df_valid[y_col].min() * 0.3
    y_max = df_valid[y_col].max() * 1.5
    ax.set_ylim(y_min, y_max)
    
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.tight_layout()
    plt.savefig(f'task_1/data/compare_filling_missed_data_linear_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
    
def get_difference_vector(df_valid: pd.DataFrame, df_restored: pd.DataFrame):
    """Вычисляет вектор ошибок для восстановленных значений"""

    result = df_valid['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'][1::2] - df_restored['Daily_new_confirmed_cases_due_to_COVID-19_(rolling_7-day_average_right-aligned)'][1::2]
    return result

def sigma_method(errors: pd.Series, number: int):
    mean = errors.mean()
    std = errors.std()
    lower = mean - number * std
    upper = mean + number * std
    outliers_mask = (errors < lower) | (errors > upper)
    return errors[outliers_mask]

def get_box_plot(errors: pd.Series):
    _, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(y=errors, ax=ax, color="skyblue")
    ax.set_title("Box-plot ошибок восстановления")
    ax.set_ylabel("Ошибка (valid - restored)")
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.tight_layout()
    plt.savefig(f'task_1/data/difference_box_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
def main():
    data_file = "task_1/DataSet1_4.csv"  # Путь к файлу с данными
    separator = ","  # Разделитель данных в файле

    data = read_data(data_file, separator)
    data["Day"] = pd.to_datetime(data["Day"])
    
    linear_data = get_linear_chart_data(data)
    get_linear_graph(linear_data)

    pie_data = get_pie_chart_data(data)
    get_pie_chart(pie_data)

    bar_data = get_bar_chart_data(data)
    get_bar_chart(bar_data)
    
    violin_data = get_violin_plot_data(data)
    get_violin_plot(violin_data)
    
    bad_data = get_bad_data(data)
    valid_data = get_valid_data(data)
    average = fill_average(bad_data.copy())
    median = fill_median(bad_data.copy())
    average_of_3 = fill_average_of_3(bad_data.copy())
    llagrange = fill_lagrange(bad_data.copy())
    cube_spline = fill_cube_spline(bad_data.copy())

    get_linear_compare(valid_data, average, median, average_of_3, llagrange, cube_spline)    
    
    difference_vector = get_difference_vector(valid_data, average_of_3)
    print(difference_vector)
    print(sigma_method(difference_vector, 3))
    print(sigma_method(difference_vector, 2))
    get_box_plot(difference_vector)

if __name__ == "__main__":
    main()
