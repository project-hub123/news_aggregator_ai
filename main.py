import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


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
LOG_FILE = "logs/user_actions.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)


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
# ЛОГИРОВАНИЕ
# ==========================================================

def log_action(action_type, input_text, result):

    log_data = {
        "timestamp": datetime.now(),
        "action": action_type,
        "input_text": str(input_text)[:300],
        "result": str(result)[:300]
    }

    if os.path.exists(LOG_FILE):
        df_logs = pd.read_csv(LOG_FILE)
        df_logs = pd.concat([df_logs, pd.DataFrame([log_data])])
    else:
        df_logs = pd.DataFrame([log_data])

    df_logs.to_csv(LOG_FILE, index=False)


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
# ПРОСТАЯ ГЕНЕРАЦИЯ САММАРИ (Extractive TF-IDF)
# ==========================================================

def generate_summary(text, num_sentences=2):

    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    sentence_scores = np.array(X.sum(axis=1)).flatten()

    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices.sort()

    summary = ". ".join([sentences[i] for i in top_indices])
    return summary


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
        "История обучения",
        "Логи действий"
    ]
)

model, vectorizer = load_model()


# ----------------------------------------------------------
# О ПРОЕКТЕ
# ----------------------------------------------------------

if menu == "О проекте":

    st.write("""
    Реализовано:
    - классификация новостей (LinearSVC, Accuracy ≈ 0.81)
    - персонализированные рекомендации
    - генерация кратких аннотаций (TF-IDF extractive)
    - логирование действий пользователя
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
        ax.set_title("Распределение новостей")
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
            log_action("predict", text_input, prediction)


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

            log_action("recommend", user_input, "top5 returned")

            csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Скачать рекомендации CSV",
                data=csv,
                file_name="recommendations.csv",
                mime="text/csv"
            )


# ----------------------------------------------------------
# ГЕНЕРАЦИЯ
# ----------------------------------------------------------

if menu == "Генерация саммари":

    text_input = st.text_area("Введите текст новости")

    if st.button("Сгенерировать краткое содержание"):
        summary = generate_summary(text_input)
        st.success(summary)
        log_action("generate_summary", text_input, summary)


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
        st.info("История обучения отсутствует")


# ----------------------------------------------------------
# ЛОГИ
# ----------------------------------------------------------

if menu == "Логи действий":

    if os.path.exists(LOG_FILE):
        df_logs = pd.read_csv(LOG_FILE)
        st.dataframe(df_logs)

        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Скачать логи CSV",
            data=csv,
            file_name="user_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("Логи пока отсутствуют")
