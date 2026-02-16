import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from config import *
from users import *
from ml_model import *
from summarizer import generate_summary
from analytics import *
from datetime import datetime


# ==========================================================
# STREAMLIT НАСТРОЙКА
# ==========================================================

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


# ==========================================================
# ЛОГИРОВАНИЕ
# ==========================================================

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


# ==========================================================
# ПЕРСОНАЛИЗАЦИЯ
# ==========================================================

def get_user_preference(username):

    if not os.path.exists(LOG_FILE):
        return None

    logs = pd.read_csv(LOG_FILE)

    if "category" not in logs.columns:
        return None

    user_logs = logs[
        (logs["user"] == username) &
        (logs["category"].notna())
    ]

    if user_logs.empty:
        return None

    return user_logs["category"].value_counts().idxmax()


def personalized_recommendations(username, df, vectorizer):

    preferred = get_user_preference(username)

    if preferred and "category" in df.columns:
        df = df[df["category"] == preferred]

    if df.empty:
        return None

    return df.sample(min(5, len(df)))


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
# ОСНОВНАЯ СИСТЕМА
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
            "Рекомендации",
            "Персональные рекомендации",
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
    # О СИСТЕМЕ
    # ======================================================

    if menu == "О системе":

        st.title(PROJECT_TITLE)

        st.markdown(f"""
**Организация:** {ORGANIZATION}  
**Разработчик:** {DEVELOPER}  
**Год разработки:** {YEAR}
""")

        st.divider()

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

    # ======================================================
    # РЕКОМЕНДАЦИИ
    # ======================================================

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

    # ======================================================
    # ПЕРСОНАЛЬНЫЕ РЕКОМЕНДАЦИИ
    # ======================================================

    if menu == "Персональные рекомендации":

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
                st.dataframe(results[["text", "category"]])
            else:
                st.info("Недостаточно данных для персональной подборки")

    # ======================================================
    # ПРЕДСКАЗАНИЕ
    # ======================================================

    if menu == "Предсказание категории":

        if model is None:
            st.warning("Модель не обучена.")
        else:
            text = st.text_area("Введите текст новости")

            if st.button("Определить категорию"):

                pred = predict_category(text, model, vectorizer)
                st.success(f"Категория: {pred}")
                log_action("predict", text, pred, pred)

    # ======================================================
    # АННОТАЦИЯ
    # ======================================================

    if menu == "Генерация аннотации":

        text = st.text_area("Введите полный текст")

        if st.button("Сформировать аннотацию"):

            summary = generate_summary(text)
            st.info(summary)
            log_action("summary", text, summary)

    # ======================================================
    # АНАЛИЗ
    # ======================================================

    if menu == "Анализ данных" and st.session_state.role in ["Аналитик", "Администратор"]:

        df = pd.read_csv("data/news_dataset.csv")

        info = dataset_overview(df)
        st.write(info)

        fig1 = plot_category_distribution(df)
        st.pyplot(fig1)

        fig2 = plot_text_length_distribution(df)
        st.pyplot(fig2)

    # ======================================================
    # ОБУЧЕНИЕ
    # ======================================================

    if menu == "Обучение модели" and st.session_state.role == "Администратор":

        df = pd.read_csv("data/news_dataset.csv")

        if st.button("Обучить модель"):

            results = train_model(df)

            st.success(f"Accuracy: {results['accuracy']:.3f}")

            cm_fig = show_confusion_matrix(
                results["confusion_matrix"],
                df["category"].unique()
            )

            st.pyplot(cm_fig)

    # ======================================================
    # ИСТОРИЯ ОБУЧЕНИЯ
    # ======================================================

    if menu == "История обучения":

        if os.path.exists(METRICS_FILE):

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

        user_edit = st.selectbox("Изменить роль", users["username"])
        new_role = st.selectbox("Новая роль",
                                ["Пользователь", "Аналитик", "Администратор"])

        if st.button("Обновить роль"):
            update_role(user_edit, new_role)
            st.success("Роль обновлена")
            st.rerun()

        user_delete = st.selectbox("Удалить пользователя",
                                   users["username"])

        if st.button("Удалить"):
            if user_delete != "admin":
                delete_user(user_delete)
                st.success("Пользователь удалён")
                st.rerun()
            else:
                st.error("Администратора удалить нельзя")
