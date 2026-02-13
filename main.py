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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ==========================================================
# НАСТРОЙКИ ПРОЕКТА
# ==========================================================

DEVELOPER = "Гаврилов Никита Дмитриевич"
PROJECT_TITLE = "Интеллектуальный агрегатор новостей"
ORGANIZATION = "ФГБУЗ МСЧ №72 ФМБА России"
YEAR = datetime.now().year

MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
USERS_FILE = "data/users.csv"
LOG_FILE = "logs/user_actions.csv"
METRICS_FILE = "models/metrics.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ==========================================================
# ХЕШИРОВАНИЕ
# ==========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ==========================================================
# СОЗДАНИЕ ADMIN
# ==========================================================

if not os.path.exists(USERS_FILE):
    pd.DataFrame([{
        "username": "admin",
        "password": hash_password("admin123"),
        "role": "Администратор"
    }]).to_csv(USERS_FILE, index=False)


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
    user = users[(users["username"] == username) &
                 (users["password"] == hashed)]
    if not user.empty:
        return user.iloc[0]["role"]
    return None

def register_user(username, password):
    users = load_users()

    if username in users["username"].values:
        return False, "Пользователь уже существует"

    new_user = {
        "username": username,
        "password": hash_password(password),
        "role": "Пользователь"
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
# АННОТАЦИЯ
# ==========================================================

def generate_summary(text, num_sentences=2):

    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if len(sentences) < 2:
        return "Введите более развернутый текст (минимум 2-3 предложения)."

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    scores = np.array(X.sum(axis=1)).flatten()
    top_indices = scores.argsort()[-num_sentences:]
    top_indices.sort()

    summary = ". ".join([sentences[i] for i in top_indices])
    return summary + "."


# ==========================================================
# ЛОГИРОВАНИЕ (с категорией)
# ==========================================================

def log_action(action_type, input_text, result, category=None):

    log_data = {
        "timestamp": datetime.now(),
        "user": st.session_state.get("username", ""),
        "action": action_type,
        "input": str(input_text)[:200],
        "result": str(result)[:200],
        "category": category
    }

    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        logs = pd.concat([logs, pd.DataFrame([log_data])])
    else:
        logs = pd.DataFrame([log_data])

    logs.to_csv(LOG_FILE, index=False)


# ==========================================================
# ОБУЧЕНИЕ МОДЕЛИ
# ==========================================================

def train_model(df):

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    metrics = pd.DataFrame([{
        "date": datetime.now(),
        "accuracy": acc
    }])

    if os.path.exists(METRICS_FILE):
        old = pd.read_csv(METRICS_FILE)
        metrics = pd.concat([old, metrics])

    metrics.to_csv(METRICS_FILE, index=False)

    return acc


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    return None, None


def predict_category(text, model, vectorizer):
    vector = vectorizer.transform([clean_text(text)])
    return model.predict(vector)[0]


# ==========================================================
# ПЕРСОНИФИКАЦИЯ
# ==========================================================

def get_user_preference(username):

    if not os.path.exists(LOG_FILE):
        return None

    logs = pd.read_csv(LOG_FILE)
    user_logs = logs[(logs["user"] == username) &
                     (logs["category"].notna())]

    if user_logs.empty:
        return None

    return user_logs["category"].value_counts().idxmax()


def personalized_recommendations(username, df):

    preferred_category = get_user_preference(username)

    if preferred_category:
        df = df[df["category"] == preferred_category]

    return df.sample(min(5, len(df)))


def recommend_news(user_text, df):

    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    news_vectors = vectorizer.fit_transform(df["clean_text"])

    user_vector = vectorizer.transform([clean_text(user_text)])
    similarities = cosine_similarity(user_vector, news_vectors).flatten()

    top_indices = similarities.argsort()[-5:][::-1]
    return df.iloc[top_indices]


# ==========================================================
# STREAMLIT
# ==========================================================

st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ==========================================================
# ВХОД / РЕГИСТРАЦИЯ
# ==========================================================

if not st.session_state.logged_in:

    st.title("Вход в систему")

    tab1, tab2 = st.tabs(["Вход", "Регистрация"])

    with tab1:
        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")

        if st.button("Войти"):
            role = login(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Неверный логин или пароль")

    with tab2:
        new_user = st.text_input("Новый логин")
        new_pass = st.text_input("Новый пароль", type="password")

        if st.button("Зарегистрироваться"):
            success, msg = register_user(new_user, new_pass)
            if success:
                st.success(msg)
            else:
                st.error(msg)


# ==========================================================
# ОСНОВНАЯ ЧАСТЬ
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
            "О системе",
            "Персональные рекомендации",
            "Рекомендации по запросу",
            "Предсказание категории",
            "Генерация аннотации",
            "Анализ данных",
            "Обучение модели",
            "История обучения",
            "Логи",
            "Управление пользователями"
        ]
    )

    model, vectorizer = load_model()


    # ======================================================
    # ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ
    # ======================================================

    if menu == "Персональные рекомендации":

        df = pd.read_csv("data/news_dataset.csv")

        results = personalized_recommendations(
            st.session_state.username, df
        )

        st.dataframe(results[["text", "category"]])


    # ======================================================
    # РЕКОМЕНДАЦИИ ПО ЗАПРОСУ
    # ======================================================

    if menu == "Рекомендации по запросу":

        df = pd.read_csv("data/news_dataset.csv")
        text = st.text_area("Введите тему")

        if st.button("Получить рекомендации"):
            results = recommend_news(text, df)
            st.dataframe(results[["text", "category"]])
            log_action("recommend_query", text, "ok")


    # ======================================================
    # ПРЕДСКАЗАНИЕ
    # ======================================================

    if menu == "Предсказание категории" and model:

        text = st.text_area("Введите текст новости")

        if st.button("Определить категорию"):
            pred = predict_category(text, model, vectorizer)
            st.success(f"Категория: {pred}")
            log_action("predict", text, pred, pred)


    # ======================================================
    # АННОТАЦИЯ
    # ======================================================

    if menu == "Генерация аннотации":

        text = st.text_area("Введите полный текст новости")

        if st.button("Сформировать аннотацию"):
            summary = generate_summary(text)
            st.info(summary)
            log_action("summary", text, summary)


    # ======================================================
    # АНАЛИЗ
    # ======================================================

    if menu == "Анализ данных" and st.session_state.role in ["Аналитик", "Администратор"]:

        df = pd.read_csv("data/news_dataset.csv")
        fig, ax = plt.subplots()
        df["category"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)


    # ======================================================
    # ОБУЧЕНИЕ
    # ======================================================

    if menu == "Обучение модели" and st.session_state.role == "Администратор":

        df = pd.read_csv("data/news_dataset.csv")

        if st.button("Обучить модель"):
            acc = train_model(df)
            st.success(f"Accuracy: {acc:.3f}")


    # ======================================================
    # ИСТОРИЯ
    # ======================================================

    if menu == "История обучения" and os.path.exists(METRICS_FILE):

        metrics = pd.read_csv(METRICS_FILE)
        st.dataframe(metrics)

        fig, ax = plt.subplots()
        ax.plot(metrics["accuracy"])
        ax.set_title("Динамика Accuracy")
        st.pyplot(fig)


    # ======================================================
    # ЛОГИ
    # ======================================================

    if menu == "Логи" and st.session_state.role in ["Аналитик", "Администратор"]:

        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE))


    # ======================================================
    # УПРАВЛЕНИЕ ПОЛЬЗОВАТЕЛЯМИ
    # ======================================================

    if menu == "Управление пользователями" and st.session_state.role == "Администратор":

        users = load_users()
        st.dataframe(users)

        user_to_edit = st.selectbox("Изменить роль", users["username"])
        new_role = st.selectbox("Новая роль", ["Пользователь", "Аналитик", "Администратор"])

        if st.button("Обновить роль"):
            update_role(user_to_edit, new_role)
            st.success("Роль обновлена")
            st.rerun()

        user_to_delete = st.selectbox("Удалить пользователя", users["username"])

        if st.button("Удалить"):
            if user_to_delete != "admin":
                delete_user(user_to_delete)
                st.success("Пользователь удалён")
                st.rerun()
            else:
                st.error("Администратора удалить нельзя")
