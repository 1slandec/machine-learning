import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Источник базы знаний: программа дисциплины «Машинное обучение».
# https://dp.hse.ru/#/summary/?implementationId=16281524547
questions = [
    "Какова основная цель дисциплины «Машинное обучение»?",
    "Какие типы задач машинного обучения рассматриваются в курсе?",
    "Какие виды обучения изучаются в рамках дисциплины?",
    "Какие языки программирования предусмотрены для анализа данных?",
    "С какими типами данных студенты учатся работать?",
    "Какие методы предобработки данных рассматриваются в курсе?",
    "Какие модели процессов анализа данных перечислены в программе?",
    "Какие алгоритмы рассматриваются в теме поиска ассоциативных правил?",
    "Какие методы изучаются для решения задач регрессии и классификации?",
    "Какие способы подбора параметров моделей указаны в программе?",
    "Какие методы рассматриваются в разделе регрессионного анализа?",
    "Какие метрики используются для оценки качества регрессионных моделей?",
    "Какие алгоритмы построения деревьев решений изучаются в курсе?",
    "Какие средства оценки качества классификации перечислены в программе?",
    "Какие методы объединения моделей рассматриваются в курсе?",
    "Какие методы кластеризации изучаются в рамках дисциплины?",
    "Какие способы представления текста рассматриваются в курсе?",
    "Какие этапы предобработки текста указаны в примере практического задания?",
    "Какие методы работы с изображениями перечислены в программе?",
    "В каком соотношении предлагается разделить данные на обучающую и тестовую выборки?"
]

answers = [
    "Сформировать знания, умения и навыки в области принципов, методов и алгоритмов машинного обучения и интеллектуального анализа данных.",
    "В курсе рассматриваются задачи регрессии, классификации, кластеризации и прогнозирования.",
    "Изучаются обучение с учителем, обучение без учителя, обучение с подкреплением и полуконтролируемое обучение.",
    "Для анализа данных и применения моделей машинного обучения предусмотрены Python или R.",
    "Студенты учатся работать с табличными, текстовыми и визуальными данными.",
    "Рассматриваются очистка, нормализация, обработка пропущенных значений, уменьшение размерности, выделение атрибутов и обнаружение выбросов.",
    "В программе перечислены KDD, SEMMA, CRISP и ASUM.",
    "В теме поиска и выявления ассоциативных правил указаны алгоритмы Apriori и SlopeOne.",
    "Для задач регрессии и классификации рассматриваются методы kNN и SVM.",
    "В программе указаны градиентный спуск и случайный поиск.",
    "Рассматриваются метод наименьших квадратов, авторегрессия, VAR, PLS, Lasso и когнитивные карты.",
    "Для оценки регрессионных моделей перечислены MAE, RMSE, MAPE и MASE.",
    "Изучаются алгоритмы CART и C4.5.",
    "В программе перечислены Accuracy, F-мера, ROC-кривая и логистическая функция потерь.",
    "Рассматриваются баггинг, бустинг и стэкинг.",
    "Изучаются k-means, EM-алгоритм и сеть Кохонена.",
    "Рассматриваются Bag of Words, TF-IDF и Word2Vec.",
    "Предлагается привести символы к одному регистру, удалить знаки препинания, специальные символы и стоп-слова, а затем выполнить лемматизацию.",
    "Перечислены скользящее окно, гистограммы ориентированных градиентов HOG, метод Далала — Триггса и представление изображения как набора визуальных слов.",
    "В примере практического задания предлагается использовать соотношение 80/20: 80% данных для обучения и 20% для тестирования."
]

test_questions = [
    # Десять вопросов, ответы на которые есть в базе.
    "Для чего изучается дисциплина «Машинное обучение»?",
    "С какими задачами машинного обучения знакомит курс?",
    "Какая предобработка данных предусмотрена программой?",
    "Какие методы используются одновременно для регрессии и классификации?",
    "Как оценивается качество регрессионных моделей и какие метрики для этого используются?",
    "Какие алгоритмы используются для построения деревьев решений?",
    "Какими способами можно объединить несколько моделей?",
    "Какие алгоритмы кластеризации проходят на занятиях?",
    "Как надо подготовить текст перед выполнением практического задания?",
    "Какую долю данных нужно оставить для обучения, а какую — для тестирования?",
    # Пять вопросов, ответов на которые в базе нет.
    "Какую видеокарту рекомендуют для выполнения практических заданий?",
    "В какой аудитории проходят лекции по машинному обучению?",
    "Какие сроки сдачи домашнего задания установлены по дисциплине?",
    "Какой размер мини-батча нужно выбирать для обучения нейросети?",
    "Какова стоимость обучения на образовательной программе?"
]

correct_indices = [0, 1, 5, 8, 11, 12, 14, 15, 17, 19, None, None, None, None, None]

# Служебные и вопросительные слова, не различающие темы запросов.
stop_words = set("""
    и в во на с со к ко у о об обо от до по за из для при через без
    а но или что чтобы как так это этот эта эти то тот та те
    я мы ты вы он она оно они мой твой наш ваш свой его ее их
    не ни нет да же ли бы быть был была было были есть будет
    все всего еще уже только также можно нужно надо перед
    чем чего чему кого кому когда где куда откуда почему зачем
    какой какая какое какие каким какими какую какого
    какова каково каковы каком каких сколько
""".split())

stemmer = SnowballStemmer("russian")


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    tokens = re.findall(r"[а-яa-z0-9]+", text)
    return " ".join(stemmer.stem(word) for word in tokens if word not in stop_words)


def jaccard_similarity(query_vector, database_vectors):
    query_words = query_vector > 0
    database_words = database_vectors > 0
    intersection = np.logical_and(query_words, database_words).sum(axis=1)
    union = np.logical_or(query_words, database_words).sum(axis=1)
    return np.divide(
        intersection, union, out=np.zeros(len(database_vectors)), where=union != 0
    )


def pearson_similarity(query_vector, database_vectors):
    scores = []
    for vector in database_vectors:
        # Для константного вектора корреляция не определена.
        if np.std(query_vector) == 0 or np.std(vector) == 0:
            scores.append(0.0)
        else:
            scores.append(np.corrcoef(query_vector, vector)[0, 1])
    return np.array(scores)


def find_answers(query, vectorizer, database_vectors):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query]).toarray()[0]
    scores = {
        "Жаккар": jaccard_similarity(query_vector, database_vectors),
        "Косинус": cosine_similarity([query_vector], database_vectors)[0],
        "Пирсон": pearson_similarity(query_vector, database_vectors),
    }
    results = {}
    for name, values in scores.items():
        # При равенстве оценок выбирается первый вопрос в базе.
        index = int(np.argmax(values))
        results[name] = {
            "index": index,
            "answer": answers[index],
            "score": float(values[index]),
        }
    return results


def evaluate(vectorizer, database_vectors):
    rows = []
    for query, correct_index in zip(test_questions, correct_indices):
        correct_answer = (
            answers[correct_index] if correct_index is not None else "Ответ отсутствует в базе"
        )
        results = find_answers(query, vectorizer, database_vectors)
        row = {"Вопрос": query, "Правильный ответ": correct_answer}
        for name, result in results.items():
            row[f"Ответ системы ({name})"] = result["answer"]
            row[f"Метрика ({name})"] = round(result["score"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    processed_questions = [preprocess_text(question) for question in questions]
    vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False)
    database_vectors = vectorizer.fit_transform(processed_questions).toarray()

    print("Количество пар вопрос-ответ:", len(questions))
    print("Размер матрицы Bag of Words:", database_vectors.shape)
    for original, processed in zip(questions[:3], processed_questions[:3]):
        print("Исходный вопрос:", original)
        print("После обработки:", processed)

    results = evaluate(vectorizer, database_vectors)
    results.to_csv(data_dir / "results.csv", index=False, encoding="utf-8-sig")
    print(results.to_string(index=False))

    print("\nПравильно найденных ответов на 10 вопросов с ответом в базе:")
    for name in ("Жаккар", "Косинус", "Пирсон"):
        correct = (
            results.loc[:9, f"Ответ системы ({name})"] == results.loc[:9, "Правильный ответ"]
        ).sum()
        print(f"{name}: {correct} из 10")


if __name__ == "__main__":
    main()
