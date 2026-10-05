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
from sklearn.linear_model import LogisticRegression


def read_data(data_file: str, separator: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame pandas."""
    df = pd.read_csv(data_file, encoding="utf-8", sep=separator)
    return df

def scaler(x_test, x_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform()
    return x_train_scaled, x_test_scaled

def ols(x_train, y_train, x_test):
    ols = LinearRegression().fit(x_train, y_train)
    return ols.predict(x_test)

def svm(x_train, y_train, x_test):
    param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 0.1, 1]
}
    _svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=0.1, probability=True, gamma='scale')).fit(x_train, y_train)
    grid = GridSearchCV(_svm, param_grid, cv=5, scoring='accuracy')
    grid.fit(x_train, y_train)

    return (_svm.predict(x_test)>=0.5).astype(int), _svm.predict_proba(x_test)

def log_regression(x_train, y_train, x_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model.predict(x_test)


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
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', label='случайный классификатор')  
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
    y_train_class = train_df['class'].values
    t_train = train_df["time_idx"]
    y_test = test_df[y_col].values
    y_test_class = test_df['class'].values
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

    x2_test_scaled, x2_train_scaled = scaler(x2_test, x2_train)

    models = {
    'kNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(probability=True, kernel='linear'),
    'LR': LogisticRegression(max_iter=1000),
    }
    y_probs = {}
    y_preds = {}
    metrics = {}

    for name, model in models.items():
        model.fit(x2_train_scaled, y_train)
        y_probs[name] = model.predict_proba(x2_test_scaled)[:,1]
        y_preds[name] = model.predict(x2_test_scaled)
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_preds[name]),
            'logloss': log_loss(y_test, y_probs[name]),
            'f1': f1_score(y_test, y_preds[name])
        }
        _confusion_matrix(y_probs[name], y_preds[name])
        roc(y_preds[name], y_probs[name], name)
    

if __name__ == "__main__":
    main()