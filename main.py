import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


# ==========================================================
# МЕТАДАННЫЕ
# ==========================================================

DEVELOPER = "Гаврилов Никита Дмитриевич"
PROJECT_TITLE = "Интеллектуальный агрегатор новостей"
ORGANIZATION = "ФГБУЗ МСЧ №72 ФМБА России"
YEAR = datetime.now().year

MODEL2_PATH = "models/model2.pkl"
VECTORIZER2_PATH = "models/vectorizer2.pkl"

METRICS_FILE = "models/metrics_history.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ==========================================================
# ПРЕДОБРАБОТКА
# ==========================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zа-я0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==========================================================
# ОБУЧЕНИЕ (MODEL2 — ОСНОВНАЯ)
# ==========================================================

def train_model(df):

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1,2),
        min_df=2
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # сохраняем модель
    joblib.dump(model, MODEL2_PATH)
    joblib.dump(vectorizer, VECTORIZER2_PATH)

    # сохраняем метрики
    metrics_data = {
        "date": datetime.now(),
        "accuracy": accuracy,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    if os.path.exists(METRICS_FILE):
        old_metrics = pd.read_csv(METRICS_FILE)
        new_metrics = pd.concat([old_metrics, pd.DataFrame([metrics_data])])
    else:
        new_metrics = pd.DataFrame([metrics_data])

    new_metrics.to_csv(METRICS_FILE, index=False)

    return accuracy, y_test, y_pred


# ==========================================================
# ПРЕДСКАЗАНИЕ КАТЕГОРИИ
# ==========================================================

def predict_category(text, model, vectorizer):
    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])
    prediction = model.predict(vector)[0]
    return prediction


# ==========================================================
# РЕКОМЕНДАЦИИ
# ==========================================================

def recommend_news(user_text, df, vectorizer):

    user_text_clean = clean_text(user_text)
    user_vector = vectorizer.transform([user_text_clean])

    df["clean_text"] = df["text"].apply(clean_text)
    news_vectors = vectorizer.transform(df["clean_text"])

    similarities = cosine_similarity(user_vector, news_vectors)
    top_indices = similarities.argsort()[0][-5:][::-1]

    return df.iloc[top_indices]


# ==========================================================
# ГЕНЕРАТИВНАЯ МОДЕЛЬ
# ==========================================================

@st.cache_resource
def load_generator():
    tokenizer = T5Tokenizer.from_pretrained("cointegrated/rut5-base")
    model = T5ForConditionalGeneration.from_pretrained("cointegrated/rut5-base")
    return tokenizer, model


def generate_summary(text):
    tokenizer, model = load_generator()
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=120, num_beams=4, early_stopping=True)

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ==========================================================
# GUI
# ==========================================================

st.set_page_config(page_title="AI Агрегатор новостей", layout="wide")

st.title(PROJECT_TITLE)
st.markdown(f"""
**Организация:** {ORGANIZATION}  
**Разработчик:** {DEVELOPER}  
**Год разработки:** {YEAR}
""")

st.divider()

menu = st.sidebar.selectbox(
    "Навигация",
    [
        "О проекте",
        "Загрузка данных",
        "Обучение модели",
        "Тестирование модели",
        "Рекомендации",
        "Генерация саммари"
    ]
)


# ----------------------------------------------------------
# ОБУЧЕНИЕ
# ----------------------------------------------------------

if menu == "Обучение модели":

    if os.path.exists("data/news_dataset.csv"):

        df = pd.read_csv("data/news_dataset.csv")

        if st.button("Обучить Model2 (LinearSVC)"):

            accuracy, y_test, y_pred = train_model(df)

            st.success(f"Accuracy: {accuracy:.3f}")
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

    else:
        st.warning("Сначала загрузите данные")


# ----------------------------------------------------------
# ТЕСТИРОВАНИЕ
# ----------------------------------------------------------

if menu == "Тестирование модели":

    if os.path.exists(MODEL2_PATH):

        model = joblib.load(MODEL2_PATH)
        vectorizer = joblib.load(VECTORIZER2_PATH)

        text_input = st.text_area("Введите текст для определения категории")

        if st.button("Определить категорию"):
            prediction = predict_category(text_input, model, vectorizer)
            st.success(f"Предсказанная категория: {prediction}")

        if os.path.exists(METRICS_FILE):
            st.subheader("История метрик")
            metrics_df = pd.read_csv(METRICS_FILE)
            st.dataframe(metrics_df)

    else:
        st.warning("Сначала обучите модель")


# ----------------------------------------------------------
# РЕКОМЕНДАЦИИ
# ----------------------------------------------------------

if menu == "Рекомендации":

    if os.path.exists(MODEL2_PATH):

        vectorizer = joblib.load(VECTORIZER2_PATH)
        df = pd.read_csv("data/news_dataset.csv")

        user_input = st.text_area("Введите тему интереса")

        if st.button("Получить рекомендации"):
            results = recommend_news(user_input, df, vectorizer)
            st.dataframe(results[["text", "category"]])

    else:
        st.warning("Сначала обучите модель")


# ----------------------------------------------------------
# ГЕНЕРАЦИЯ
# ----------------------------------------------------------

if menu == "Генерация саммари":

    text_input = st.text_area("Введите текст новости")

    if st.button("Сгенерировать"):
        summary = generate_summary(text_input)
        st.success(summary)
