import re
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

TARGET_CLASS = "Конкурсы и розыгрыши"
RANDOM_STATE = 42
nlp = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
stemmer = SnowballStemmer("russian")

def read_data(data_file):
    df = pd.read_csv(data_file, encoding="utf-8")
    df = df[["Text", "TClass"]].rename(columns={"Text": "text", "TClass": "label"})
    df = df.dropna().copy()
    df["target"] = (df["label"] == TARGET_CLASS).astype(int)
    return df.drop_duplicates(["text", "target"]).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower().replace("ё", "е").replace("\\n", " ")
    return re.sub(r"[^а-яa-z\s]", " ", text)

def preprocess_stemming(text):
    tokens = nlp.make_doc(clean_text(text))
    return " ".join(
        stemmer.stem(token.text)
        for token in tokens if token.is_alpha and not token.is_stop
    )

def preprocess_lemmatization(text):
    doc = nlp(clean_text(text))
    return " ".join(
        token.lemma_.replace("ё", "е")
        for token in doc
        if token.is_alpha and not token.is_stop and not nlp.vocab[token.lemma_].is_stop
    )

def prepare_data(df):
    df = df.copy()
    df["clean_stem"] = df["text"].apply(preprocess_stemming)
    df["clean_lemma"] = df["text"].apply(preprocess_lemmatization)
    df = df.loc[df["clean_stem"].ne("") & df["clean_lemma"].ne("")].copy()
    conflict_stem = df.groupby("clean_stem")["target"].transform("nunique") > 1
    conflict_lemma = df.groupby("clean_lemma")["target"].transform("nunique") > 1
    df = df.loc[~conflict_stem & ~conflict_lemma]
    return df.drop_duplicates("clean_stem").drop_duplicates("clean_lemma").reset_index(drop=True)

def train_models(X_train, X_test, y_train, y_test):
    models = {
        "kNN": (KNeighborsClassifier(metric="cosine", algorithm="brute"),
                {"model__n_neighbors": [3, 5, 7]}),
        "SVM": (SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
                {"model__C": [0.01, 0.1, 1.0, 10.0]}),
        "LR": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
               {"model__C": [0.01, 0.1, 1.0, 10.0]}),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, scores = [], {}
    for name, (model, params) in models.items():
        pipeline = Pipeline([
            ("vectorizer", CountVectorizer(max_features=3000)), ("model", model)
        ])
        grid = GridSearchCV(pipeline, params, cv=cv, scoring="f1", n_jobs=1)
        grid.fit(X_train, y_train)
        prediction = grid.predict(X_test)
        scores[name] = grid.predict_proba(X_test)[:, 1]
        rows.append({
            "Модель": name,
            "Accuracy": accuracy_score(y_test, prediction),
            "F1": f1_score(y_test, prediction, zero_division=0),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, scores[name]),
            "CV_F1": grid.best_score_,
            "Параметры": str(grid.best_params_),
        })
    return pd.DataFrame(rows), scores

def plot_roc(y_test, scores, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        area = roc_auc_score(y_test, score)
        ax.plot(fpr, tpr, label=f"{name}: AUC = {area:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    df = prepare_data(read_data("SocialMedia.csv"))
    print("Размер подготовленной выборки:", df.shape[0])
    print(df["target"].value_counts().sort_index())

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE
    )
    print(f"Train: {len(train)}, test: {len(test)}")

    metrics_stem, scores_stem = train_models(
        train["clean_stem"], test["clean_stem"], train["target"], test["target"]
    )
    plot_roc(test["target"], scores_stem, "Стемминг + BoW", "task_6/data/roc_stemming.png")

    metrics_lemma, scores_lemma = train_models(
        train["clean_lemma"], test["clean_lemma"], train["target"], test["target"]
    )
    plot_roc(test["target"], scores_lemma, "Лемматизация + BoW", "task_6/data/roc_lemmatization.png")
    comparison = pd.concat([
        metrics_stem.assign(Обработка="Стемминг"),
        metrics_lemma.assign(Обработка="Лемматизация"),
    ], ignore_index=True)
    comparison.to_csv("task_6/data/metrics.csv", index=False, encoding="utf-8-sig")
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
