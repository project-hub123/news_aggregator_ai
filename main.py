import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime
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

MODEL_PATH = "models/model2.pkl"
VECTORIZER_PATH = "models/vectorizer2.pkl"
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
# ЗАГРУЗКА МОДЕЛИ
# ==========================================================

@st.cache_resource
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
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=120,
            num_beams=4,
            early_stopping=True
        )

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
        "Анализ данных",
        "Предсказание категории",
        "Рекомендации",
        "Генерация саммари",
        "История обучения"
    ]
)

model, vectorizer = load_model()


# ----------------------------------------------------------
# О ПРОЕКТЕ
# ----------------------------------------------------------

if menu == "О проекте":

    st.write("""
    Сервис реализует:
    - классификацию новостей (LinearSVC, Accuracy ≈ 0.81)
    - персонализированные рекомендации
    - генерацию кратких аннотаций
    - анализ распределения данных
    """)


# ----------------------------------------------------------
# АНАЛИЗ ДАННЫХ
# ----------------------------------------------------------

if menu == "Анализ данных":

    if os.path.exists("data/news_dataset.csv"):

        df = pd.read_csv("data/news_dataset.csv")

        st.subheader("Распределение категорий")

        fig, ax = plt.subplots(figsize=(10,5))
        df["category"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Распределение новостей по категориям")
        ax.set_ylabel("Количество")
        st.pyplot(fig)

        st.write("Размер датасета:", df.shape)

    else:
        st.warning("Файл news_dataset.csv не найден")


# ----------------------------------------------------------
# ПРЕДСКАЗАНИЕ
# ----------------------------------------------------------

if menu == "Предсказание категории":

    if model is None:
        st.warning("Модель не найдена. Обучите её в ноутбуке.")
    else:
        text_input = st.text_area("Введите текст новости")

        if st.button("Определить категорию"):
            prediction = predict_category(text_input, model, vectorizer)
            st.success(f"Предсказанная категория: {prediction}")


# ----------------------------------------------------------
# РЕКОМЕНДАЦИИ
# ----------------------------------------------------------

if menu == "Рекомендации":

    if model is None:
        st.warning("Модель не найдена.")
    else:
        df = pd.read_csv("data/news_dataset.csv")
        user_input = st.text_area("Введите интересующую тему")

        if st.button("Получить рекомендации"):
            results = recommend_news(user_input, df, vectorizer)
            st.dataframe(results[["text", "category"]])


# ----------------------------------------------------------
# ГЕНЕРАЦИЯ
# ----------------------------------------------------------

if menu == "Генерация саммари":

    text_input = st.text_area("Введите текст новости")

    if st.button("Сгенерировать краткое содержание"):
        summary = generate_summary(text_input)
        st.success(summary)


# ----------------------------------------------------------
# ИСТОРИЯ ОБУЧЕНИЯ
# ----------------------------------------------------------

if menu == "История обучения":

    if os.path.exists(METRICS_FILE):

        metrics_df = pd.read_csv(METRICS_FILE)
        st.dataframe(metrics_df)

        fig, ax = plt.subplots()
        ax.plot(metrics_df["accuracy"])
        ax.set_title("Динамика Accuracy")
        ax.set_ylabel("Accuracy")
        st.pyplot(fig)

    else:
        st.info("История обучения пока отсутствует")
