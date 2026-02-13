import pandas as pd

# Загружаем датасет
df = pd.read_csv("data/russian_news_2020.csv")

print("Колонки датасета:")
print(df.columns)

# Объединяем заголовок и текст
df["text_full"] = df["title"].astype(str) + " " + df["text"].astype(str)

# Используем rubric как категорию
df_final = df[["text_full", "rubric"]].copy()
df_final.columns = ["text", "category"]

# Удаляем пустые значения
df_final.dropna(inplace=True)

print("Исходный размер:", df_final.shape)

# Если данных больше 10000 — уменьшаем
if len(df_final) > 10000:
    df_final = df_final.sample(n=10000, random_state=42)
    print("После уменьшения:", df_final.shape)
else:
    print("Выборка не уменьшалась.")

# Сохраняем
df_final.to_csv("data/news_dataset.csv", index=False)

print("Готово. Финальный размер:", df_final.shape)
