import pandas as pd

# Загружаем датасет
df = pd.read_csv("data/russian_news.csv")

print("Колонки датасета:")
print(df.columns)

# Объединяем заголовок и текст
df["text_full"] = df["title"].astype(str) + " " + df["text"].astype(str)

# Используем rubric как категорию
df_final = df[["text_full", "rubric"]].copy()
df_final.columns = ["text", "category"]

# Удаляем пустые значения
df_final.dropna(inplace=True)

# Можно уменьшить размер (например до 10000 строк для скорости)
df_final = df_final.sample(n=10000, random_state=42)

# Сохраняем подготовленный датасет
df_final.to_csv("data/news_dataset.csv", index=False)

print("Готово. Размер:", df_final.shape)
