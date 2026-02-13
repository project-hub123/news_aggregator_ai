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
from sklearn.svm import LinearSVC


# ==========================================================
# НАСТРОЙКИ
# ==========================================================

DEVELOPER = "Гаврилов Никита Дмитриевич"
PROJECT_TITLE = "Интеллектуальный агрегатор новостей"
ORGANIZATION = "ФГБУЗ МСЧ №72 ФМБА России"
YEAR = datetime.now().year

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
USERS_FILE = "data/users.csv"
LOG_FILE = "logs/user_actions.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ==========================================================
# СОЗДАНИЕ ПОЛЬЗОВАТЕЛЕЙ ПО УМОЛЧАНИЮ
# ==========================================================

if not os.path.exists(USERS_FILE):
    users_df = pd.DataFrame([
        {"username": "admin", "password": "admin123", "role": "Администратор"},
        {"username": "analyst", "password": "analyst123", "role": "Аналитик"},
        {"username": "user", "password": "user123", "role": "Пользователь"}
    ])
    users_df.to_csv(USERS_FILE, index=False)


# ==========================================================
# АВТОРИЗАЦИЯ
# ==========================================================

def login(username, password):
    users = pd.read_csv(USERS_FILE)
    user = users[
        (users["username"] == username) &
        (users["password"] == password)
    ]
    if not user.empty:
        return user.iloc[0]["role"]
    return None


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
# ЛОГИ
# ==========================================================

def log_action(action_type, input_text, result):
    log_data = {
        "timestamp": datetime.now(),
        "user": st.session_state.get("username", ""),
        "action": action_type,
        "input_text": str(input_text)[:200],
        "result": str(result)[:200]
    }

    if os.path.exists(LOG_FILE):
        df_logs = pd.read_csv(LOG_FILE)
        df_logs = pd.concat([df_logs, pd.DataFrame([log_data])])
    else:
        df_logs = pd.DataFrame([log_data])

    df_logs.to_csv(LOG_FILE, index=False)


# ==========================================================
# МОДЕЛЬ
# ==========================================================

def train_model(df):
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["category"]

    model = LinearSVC()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def predict_category(text, model, vectorizer):
    text_clean = clean_text(text)
    vector = vectorizer.transform([text_clean])
    return model.predict(vector)[0]


def recommend_news(user_text, df):
    df["clean_text"] = df["text"].apply(clean_text)

    rec_vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    news_vectors = rec_vectorizer.fit_transform(df["clean_text"])

    user_vector = rec_vectorizer.transform([clean_text(user_text)])
    similarities = cosine_similarity(user_vector, news_vectors).flatten()

    top_indices = similarities.argsort()[-5:][::-1]
    return df.iloc[top_indices]


# ==========================================================
# GUI
# ==========================================================

st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

# ---------- Авторизация ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.title("Вход в систему")

    username = st.text_input("Логин")
    password = st.text_input("Пароль", type="password")

    if st.button("Войти"):
        role = login(username, password)
        if role:
            st.session_state.logged_in = True
            st.session_state.role = role
            st.session_state.username = username
            st.success("Успешный вход")
            st.rerun()
        else:
            st.error("Неверный логин или пароль")

else:

    st.sidebar.write(f"Пользователь: {st.session_state.username}")
    st.sidebar.write(f"Роль: {st.session_state.role}")

    if st.sidebar.button("Выйти"):
        st.session_state.logged_in = False
        st.rerun()

    st.title(PROJECT_TITLE)
    st.markdown(f"""
    **Организация:** {ORGANIZATION}  
    **Разработчик:** {DEVELOPER}  
    **Год разработки:** {YEAR}
    """)

    menu = st.sidebar.selectbox(
        "Навигация",
        [
            "Рекомендации",
            "Предсказание категории",
            "Генерация саммари",
            "Анализ данных",
            "Обучение модели",
            "Логи"
        ]
    )

    model, vectorizer = load_model()

    # ---------- Рекомендации ----------
    if menu == "Рекомендации":

        df = pd.read_csv("data/news_dataset.csv")

        text = st.text_area("Введите тему")

        if st.button("Получить рекомендации"):
            results = recommend_news(text, df)
            st.dataframe(results[["text", "category"]])
            log_action("recommend", text, "ok")

    # ---------- Предсказание ----------
    if menu == "Предсказание категории":

        if model is None:
            st.warning("Модель не обучена")
        else:
            text = st.text_area("Введите текст")

            if st.button("Определить"):
                pred = predict_category(text, model, vectorizer)
                st.success(pred)
                log_action("predict", text, pred)

    # ---------- Саммари ----------
    if menu == "Генерация саммари":

        text = st.text_area("Введите текст")

        if st.button("Сгенерировать"):
            st.info(text[:300])
            log_action("summary", text, "ok")

    # ---------- Аналитика (только аналитик и админ) ----------
    if menu == "Анализ данных" and st.session_state.role in ["Аналитик", "Администратор"]:

        df = pd.read_csv("data/news_dataset.csv")

        fig, ax = plt.subplots()
        df["category"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # ---------- Обучение (только админ) ----------
    if menu == "Обучение модели" and st.session_state.role == "Администратор":

        df = pd.read_csv("data/news_dataset.csv")

        if st.button("Обучить модель"):
            train_model(df)
            st.success("Модель обучена")

    # ---------- Логи (аналитик и админ) ----------
    if menu == "Логи" and st.session_state.role in ["Аналитик", "Администратор"]:

        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE))
        else:
            st.info("Логи отсутствуют")
