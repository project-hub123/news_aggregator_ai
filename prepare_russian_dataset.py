import pandas as pd

# Загружаем оригинальный датасет
df = pd.read_csv("data/russian_news.csv")

# Проверяем названия колонок
print(df.columns)

# ОБЪЕДИНЯЕМ ЗАГОЛОВОК И ТЕКСТ (если есть title)
if "title" in df.columns and "text" in df.columns:
    df["text_full"] = df["title"] + " " + df["text"]
    text_column = "text_full"
elif "text" in df.columns:
    text_column = "text"
else:
    raise Exception("Не найдена колонка с текстом")

# Определяем колонку категории
if "topic" in df.columns:
    category_column = "topic"
elif "category" in df.columns:
    category_column = "category"
else:
    raise Exception("Не найдена колонка категории")

# Оставляем только нужные столбцы
df_final = df[[text_column, category_column]].copy()
df_final.columns = ["text", "category"]

# Удаляем пустые значения
df_final.dropna(inplace=True)

# Сохраняем в формат для модели
df_final.to_csv("data/news_dataset.csv", index=False)

print("Датасет подготовлен. Размер:", df_final.shape)
