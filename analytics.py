import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def dataset_overview(df):

    info = {
        "Размер датасета": df.shape,
        "Количество категорий": df["category"].nunique() if "category" in df.columns else 0,
        "Средняя длина текста": df["text"].str.len().mean() if "text" in df.columns else 0
    }

    return info


def plot_category_distribution(df):

    if "category" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    df["category"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Распределение новостей по категориям")
    ax.set_xlabel("Категория")
    ax.set_ylabel("Количество")

    plt.xticks(rotation=45)

    return fig


def plot_text_length_distribution(df):

    if "text" not in df.columns:
        return None

    df["text_length"] = df["text"].str.len()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(df["text_length"], bins=50, ax=ax)

    ax.set_title("Распределение длины текстов")
    ax.set_xlabel("Длина текста")
    ax.set_ylabel("Частота")

    return fig


def show_confusion_matrix(cm, labels):

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истинное значение")
    ax.set_title("Матрица ошибок")

    return fig
