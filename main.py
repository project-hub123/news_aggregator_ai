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
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


# ==========================================================
# МЕТАДАННЫЕ
# ==========================================================

DEVELOPER = "Гаврилов Никита Дмитриевич"
PROJECT_TITLE = "Интеллектуальный агрегатор новостей"
ORGANIZATION = "ФГБУЗ МСЧ №72 ФМБА России"
YEAR = datetime.now().year

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
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
    text = re.sub(r"[^а-яa-z0-9\s]", " ", text)
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
# ОБУЧЕНИЕ МОДЕЛИ
# ==========================================================

def train_model(df):

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(df["clean_text"])
    y = df["category"]

    model = LinearSVC()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer


# ==========================================================
# ПРЕДСКАЗАНИЕ
# ==========================================================

def predict_category(text, model, vectorizer):
    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])
    prediction = model.predict(vector)[0]
    return prediction


# ==========================================================
# РЕКОМЕНДАЦИИ (исправленные)
# ==========================================================

def recommend_news(user_text, df):

    df["clean_text"] = df["text"].apply(clean_text)

    rec_vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,2)
    )

    news_vectors = rec_vectorizer.fit_transform(df["clean_text"])

    user_text_clean = clean_text(user_text)
    user_vector = rec_vectorizer.transform([user_text_clean])

    similarities = cosine_similarity(user_vector, news_vectors).flatten()

    top_indices = similarities.argsort()[-5:][::-1]

    return df.iloc[top_indices]


# ==========================================================
# ГЕНЕРАЦИЯ САММАРИ
# ==========================================================

def generate_summary(text, num_sentences=2):

    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    if len(sentences) < 2:
        return "Введите более развернутый текст (минимум 2-3 предложения)."

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    sentence_scores = np.array(X.sum(axis=1)).flatten()

    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices.sort()

    summary = ". ".join([sentences[i] for i in top_indices])

    return summary + "."


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
        "Обучение модели",
        "Анализ данных",
        "Предсказание категории",
        "Рекомендации",
        "Генерация саммари",
        "Логи действий"
    ]
)

model, vectorizer = load_model()


# ==========================================================
# О ПРОЕКТЕ
# ==========================================================

if menu == "О проекте":

    st.write("""
    Сервис выполняет:
    - классификацию новостей
    - персонализированные рекомендации
    - автоматическую генерацию краткого содержания
    - логирование действий пользователя
    """)


# ==========================================================
# ОБУЧЕНИЕ
# ==========================================================

if menu == "Обучение модели":

    if os.path.exists("data/news_dataset.csv"):

        df = pd.read_csv("data/news_dataset.csv")

        if st.button("Обучить модель"):

            model, vectorizer = train_model(df)
            st.success("Модель успешно обучена и сохранена.")

    else:
        st.warning("Файл data/news_dataset.csv не найден.")


# ==========================================================
# АНАЛИЗ
# ==========================================================

if menu == "Анализ данных":

    if os.path.exists("data/news_dataset.csv"):

        df = pd.read_csv("data/news_dataset.csv")

        fig, ax = plt.subplots(figsize=(10,5))
        df["category"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Распределение категорий")
        st.pyplot(fig)

        st.write("Размер датасета:", df.shape)

    else:
        st.warning("Файл news_dataset.csv не найден.")


# ==========================================================
# ПРЕДСКАЗАНИЕ
# ==========================================================

if menu == "Предсказание категории":

    if model is None:
        st.warning("Сначала обучите модель.")
    else:
        text_input = st.text_area("Введите текст новости")

        if st.button("Определить категорию"):
            prediction = predict_category(text_input, model, vectorizer)
            st.success(f"Категория: {prediction}")
            log_action("predict", text_input, prediction)


# ==========================================================
# РЕКОМЕНДАЦИИ
# ==========================================================

if menu == "Рекомендации":

    if os.path.exists("data/news_dataset.csv"):

        df = pd.read_csv("data/news_dataset.csv")

        user_input = st.text_area("Введите интересующую тему")

        if st.button("Получить рекомендации"):

            results = recommend_news(user_input, df)

            st.dataframe(results[["text", "category"]])

            log_action("recommend", user_input, "top5")

    else:
        st.warning("Файл news_dataset.csv не найден.")


# ==========================================================
# САММАРИ
# ==========================================================

if menu == "Генерация саммари":

    text_input = st.text_area("Введите текст новости")

    if st.button("Сгенерировать краткое содержание"):
        summary = generate_summary(text_input)
        st.info(summary)
        log_action("summary", text_input, summary)


# ==========================================================
# ЛОГИ
# ==========================================================

if menu == "Логи действий":

    if os.path.exists(LOG_FILE):
        df_logs = pd.read_csv(LOG_FILE)
        st.dataframe(df_logs)
    else:
        st.info("Логи отсутствуют.")
