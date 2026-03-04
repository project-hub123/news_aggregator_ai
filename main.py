import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import *
from users import *
from ml_model import *
from summarizer import generate_summary
from analytics import *
from datetime import datetime


st.set_page_config(
    page_title=PROJECT_TITLE,
    layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "role" not in st.session_state:
    st.session_state.role = None


def log_action(action, input_text="", result="", category=None):

    log_data = {
        "timestamp": datetime.now(),
        "user": st.session_state.username,
        "action": action,
        "input": str(input_text)[:300],
        "result": str(result)[:300],
        "category": category
    }

    if os.path.exists(LOG_FILE):
        logs = pd.read_csv(LOG_FILE)
        logs = pd.concat([logs, pd.DataFrame([log_data])], ignore_index=True)
    else:
        logs = pd.DataFrame([log_data])

    logs.to_csv(LOG_FILE, index=False)


def personalized_recommendations(username, df, vectorizer):

    if not os.path.exists(LOG_FILE):
        return df.sample(min(15, len(df)))

    logs = pd.read_csv(LOG_FILE)

    user_logs = logs[logs["user"] == username]

    if user_logs.empty:
        return df.sample(min(15, len(df)))

    history_text = user_logs["input"].dropna().astype(str)

    if len(history_text) == 0:
        return df.sample(min(15, len(df)))

    df = df.copy()

    df["clean_text"] = df["text"].apply(clean_text)

    news_vectors = vectorizer.transform(df["clean_text"])

    user_vectors = vectorizer.transform(history_text)

    user_profile = np.asarray(user_vectors.mean(axis=0))

    similarity = cosine_similarity(user_profile, news_vectors).flatten()

    top_indices = similarity.argsort()[-15:][::-1]

    return df.iloc[top_indices]


if not st.session_state.logged_in:

    st.title("Вход в систему")

    tab1 = st.tabs(["Вход"])[0]

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

else:

    st.sidebar.write(f"Пользователь: {st.session_state.username}")
    st.sidebar.write(f"Роль: {st.session_state.role}")

    if st.sidebar.button("Выйти"):
        st.session_state.logged_in = False
        st.rerun()

    menu_items = [
        "О системе",
        "Рекомендации",
        "Персональные рекомендации",
        "Генерация аннотации"
    ]

    if st.session_state.role in ["Аналитик", "Администратор"]:
        menu_items.append("Анализ данных")
        menu_items.append("Логи")

    if st.session_state.role == "Администратор":
        menu_items.append("Предсказание категории")
        menu_items.append("Обучение модели")
        menu_items.append("История обучения")
        menu_items.append("Управление пользователями")

    menu = st.sidebar.selectbox(
        "Навигация",
        menu_items
    )

    model, vectorizer = load_model()

    if menu == "О системе":

        st.title(PROJECT_TITLE)

        st.markdown(f"""
**Организация:** {ORGANIZATION}  
**Разработчик:** {DEVELOPER}  
**Год разработки:** {YEAR}
""")

        st.subheader("Назначение системы")

        st.write("""
Интеллектуальный агрегатор новостей предназначен для автоматизированной
обработки текстовой новостной информации. Система анализирует текст,
определяет его тематику, формирует рекомендации и создает краткие аннотации.

Решение реализовано на основе методов машинного обучения и
использует TF-IDF векторизацию, классификацию LinearSVC
и косинусную меру близости.
""")

        st.subheader("Основные функции")

        st.write("""
Классификация новостей по рубрикам  
Формирование рекомендаций  
Персонализированная выдача  
Генерация аннотаций  
Анализ датасета  
Обучение модели  
Журналирование действий  
""")

        st.subheader("Роли пользователей")

        st.write("""
Пользователь — базовый доступ  
Аналитик — анализ данных и журналов  
Администратор — управление пользователями и обучение модели  
""")

    if menu == "Рекомендации":

        df = pd.read_csv("data/news_dataset.csv")

        user_text = st.text_area("Введите тему")

        if st.button("Получить рекомендации"):

            if vectorizer is None:
                st.warning("Модель не обучена.")
            else:
                results = recommend_news(user_text, df, vectorizer)
                st.dataframe(results[["text", "category"]])
                log_action("recommend", user_text, "ok")

    if menu == "Персональные рекомендации":

        st.subheader("Персональные рекомендации")

        st.write("""
Персональные рекомендации формируются на основе анализа истории действий пользователя.

Алгоритм работы:
1. Из журнала логов извлекаются тексты запросов пользователя.
2. На их основе формируется TF-IDF профиль интересов пользователя.
3. Для всех новостей вычисляется косинусная мера сходства с этим профилем.
4. Пользователю предлагаются новости с наибольшей близостью.

Таким образом система учитывает реальные интересы пользователя.
""")

        df = pd.read_csv("data/news_dataset.csv")

        if vectorizer is None:
            st.warning("Модель не обучена.")
        else:
            results = personalized_recommendations(
                st.session_state.username,
                df,
                vectorizer
            )

            if results is not None:
                st.write("Рекомендованные новости:")
                st.dataframe(results[["text", "category"]])
            else:
                st.info("Недостаточно данных для персональной подборки")

    if menu == "Предсказание категории":

        if model is None:
            st.warning("Модель не обучена.")
        else:
            text = st.text_area("Введите текст новости")

            if st.button("Определить категорию"):

                pred = predict_category(text, model, vectorizer)
                st.success(f"Категория: {pred}")
                log_action("predict", text, pred, pred)

    if menu == "Генерация аннотации":

        text = st.text_area("Введите полный текст")

        if st.button("Сформировать аннотацию"):

            summary = generate_summary(text)
            st.info(summary)
            log_action("summary", text, summary)

    if menu == "Анализ данных":

        df = pd.read_csv("data/news_dataset.csv")

        info = dataset_overview(df)
        st.write(info)

        fig1 = plot_category_distribution(df)
        st.pyplot(fig1)

        fig2 = plot_text_length_distribution(df)
        st.pyplot(fig2)

    if menu == "Обучение модели":

        df = pd.read_csv("data/news_dataset.csv")

        if st.button("Обучить модель"):

            results = train_model(df)

            st.success(f"Accuracy: {results['accuracy']:.3f}")

            cm_fig = show_confusion_matrix(
                results["confusion_matrix"],
                df["category"].unique()
            )

            st.pyplot(cm_fig)

    if menu == "История обучения":

        if os.path.exists(METRICS_FILE):

            metrics = pd.read_csv(METRICS_FILE)

            st.dataframe(metrics)

            fig, ax = plt.subplots()
            ax.plot(metrics["accuracy"])
            ax.set_title("Динамика Accuracy")
            st.pyplot(fig)

    if menu == "Логи":

        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE))

    if menu == "Управление пользователями":

        users = load_users()

        st.dataframe(users)

        user_edit = st.selectbox("Изменить роль", users["username"])
        new_role = st.selectbox(
            "Новая роль",
            ["Пользователь", "Аналитик", "Администратор"]
        )

        if st.button("Обновить роль"):
            update_role(user_edit, new_role)
            st.success("Роль обновлена")
            st.rerun()

        user_delete = st.selectbox("Удалить пользователя", users["username"])

        if st.button("Удалить"):
            if user_delete != "admin":
                delete_user(user_delete)
                st.success("Пользователь удалён")
                st.rerun()
            else:
                st.error("Администратора удалить нельзя")