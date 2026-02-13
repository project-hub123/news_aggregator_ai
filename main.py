import os
import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import hashlib

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
# ХЕШ ПАРОЛЯ
# ==========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ==========================================================
# СОЗДАНИЕ ADMIN ПО УМОЛЧАНИЮ
# ==========================================================

if not os.path.exists(USERS_FILE):
    users_df = pd.DataFrame([
        {
            "username": "admin",
            "password": hash_password("admin123"),
            "role": "Администратор"
        }
    ])
    users_df.to_csv(USERS_FILE, index=False)


# ==========================================================
# ПОЛЬЗОВАТЕЛИ
# ==========================================================

def load_users():
    return pd.read_csv(USERS_FILE)


def save_users(df):
    df.to_csv(USERS_FILE, index=False)


def login(username, password):
    users = load_users()
    hashed = hash_password(password)
    user = users[
        (users["username"] == username) &
        (users["password"] == hashed)
    ]
    if not user.empty:
        return user.iloc[0]["role"]
    return None


def register_user(username, password, role="Пользователь"):
    users = load_users()

    if username in users["username"].values:
        return False, "Пользователь уже существует"

    new_user = {
        "username": username,
        "password": hash_password(password),
        "role": role
    }

    users = pd.concat([users, pd.DataFrame([new_user])])
    save_users(users)

    return True, "Регистрация успешна"


def delete_user(username):
    users = load_users()
    users = users[users["username"] != username]
    save_users(users)


def update_role(username, new_role):
    users = load_users()
    users.loc[users["username"] == username, "role"] = new_role
    save_users(users)


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
    vector = vectorizer.transform([clean_text(text)])
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

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ==========================================================
# АВТОРИЗАЦИЯ / РЕГИСТРАЦИЯ
# ==========================================================

if not st.session_state.logged_in:

    st.title("Вход / Регистрация")

    tab1, tab2 = st.tabs(["Вход", "Регистрация"])

    with tab1:
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

    with tab2:
        new_user = st.text_input("Новый логин")
        new_pass = st.text_input("Новый пароль", type="password")

        if st.button("Зарегистрироваться"):
            success, message = register_user(new_user, new_pass)
            if success:
                st.success(message)
            else:
                st.error(message)


# ==========================================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# ==========================================================

else:

    st.sidebar.write(f"Пользователь: {st.session_state.username}")
    st.sidebar.write(f"Роль: {st.session_state.role}")

    if st.sidebar.button("Выйти"):
        st.session_state.logged_in = False
        st.rerun()

    menu = st.sidebar.selectbox(
        "Навигация",
        [
            "Рекомендации",
            "Предсказание категории",
            "Анализ данных",
            "Обучение модели",
            "Логи",
            "Управление пользователями"
        ]
    )

    model, vectorizer = load_model()

    # =============================
    # РЕКОМЕНДАЦИИ
    # =============================
    if menu == "Рекомендации":
        df = pd.read_csv("data/news_dataset.csv")
        text = st.text_area("Введите тему")

        if st.button("Получить рекомендации"):
            results = recommend_news(text, df)
            st.dataframe(results[["text", "category"]])
            log_action("recommend", text, "ok")

    # =============================
    # ПРЕДСКАЗАНИЕ
    # =============================
    if menu == "Предсказание категории" and model:
        text = st.text_area("Введите текст")

        if st.button("Определить"):
            pred = predict_category(text, model, vectorizer)
            st.success(pred)
            log_action("predict", text, pred)

    # =============================
    # АНАЛИЗ
    # =============================
    if menu == "Анализ данных" and st.session_state.role in ["Аналитик", "Администратор"]:
        df = pd.read_csv("data/news_dataset.csv")
        fig, ax = plt.subplots()
        df["category"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # =============================
    # ОБУЧЕНИЕ
    # =============================
    if menu == "Обучение модели" and st.session_state.role == "Администратор":
        df = pd.read_csv("data/news_dataset.csv")
        if st.button("Обучить"):
            train_model(df)
            st.success("Модель обучена")

    # =============================
    # ЛОГИ
    # =============================
    if menu == "Логи" and st.session_state.role in ["Аналитик", "Администратор"]:
        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE))

    # =============================
    # УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ
    # =============================
    if menu == "Управление пользователями" and st.session_state.role == "Администратор":

        users = load_users()
        st.dataframe(users)

        st.subheader("Изменить роль")

        user_to_edit = st.selectbox("Выберите пользователя", users["username"])
        new_role = st.selectbox("Новая роль", ["Пользователь", "Аналитик", "Администратор"])

        if st.button("Обновить роль"):
            update_role(user_to_edit, new_role)
            st.success("Роль обновлена")
            st.rerun()

        st.subheader("Удалить пользователя")

        user_to_delete = st.selectbox("Удалить пользователя", users["username"])

        if st.button("Удалить"):
            if user_to_delete != "admin":
                delete_user(user_to_delete)
                st.success("Пользователь удалён")
                st.rerun()
            else:
                st.error("Админа удалить нельзя")
