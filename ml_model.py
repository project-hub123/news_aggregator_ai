import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.feature_extraction.text import TfidfVectorizer

from config import MODEL_PATH, VECTORIZER_PATH, METRICS_FILE


# ==========================================================
# ПРЕДОБРАБОТКА
# ==========================================================

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# ОБУЧЕНИЕ МОДЕЛИ
# ==========================================================

def train_model(df):

    if "text" not in df.columns or "category" not in df.columns:
        raise Exception("В датасете должны быть колонки text и category")

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    cm = confusion_matrix(y_test, y_pred)

    # Сохраняем модель
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    # Сохраняем метрики
    metrics_row = pd.DataFrame([{
        "date": datetime.now(),
        "accuracy": acc
    }])

    if os.path.exists(METRICS_FILE):
        old = pd.read_csv(METRICS_FILE)
        metrics_row = pd.concat([old, metrics_row])

    metrics_row.to_csv(METRICS_FILE, index=False)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm
    }


# ==========================================================
# ЗАГРУЗКА МОДЕЛИ
# ==========================================================

def load_model():

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer

    return None, None


# ==========================================================
# ПРЕДСКАЗАНИЕ
# ==========================================================

def predict_category(text, model, vectorizer):

    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])
    prediction = model.predict(vector)[0]

    return prediction


# ==========================================================
# РЕКОМЕНДАЦИИ ПО КОСИНУСНОЙ БЛИЗОСТИ
# ==========================================================

from sklearn.metrics.pairwise import cosine_similarity

def recommend_news(user_text, df, vectorizer):

    df["clean_text"] = df["text"].apply(clean_text)

    news_vectors = vectorizer.transform(df["clean_text"])

    user_vector = vectorizer.transform([clean_text(user_text)])

    similarities = cosine_similarity(user_vector, news_vectors).flatten()

    top_indices = similarities.argsort()[-5:][::-1]

    return df.iloc[top_indices]
