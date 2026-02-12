import os
import pandas as pd

# ================================
# НАСТРОЙКИ
# ================================

INPUT_FILE = "news-article-categories.csv"  # <-- ваше реальное имя файла
OUTPUT_DIR = "data"
OUTPUT_FILE = "news_dataset.csv"


# ================================
# СОЗДАНИЕ ПАПКИ data
# ================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================
# ЗАГРУЗКА ДАННЫХ
# ================================

print("Загрузка исходного датасета...")

try:
    df = pd.read_csv(INPUT_FILE)
except Exception as e:
    print("Ошибка при чтении файла:", e)
    exit()

print("Данные успешно загружены.")
print(f"Количество строк: {len(df)}")
print(f"Колонки:", df.columns.tolist())


# ================================
# ОБЪЕДИНЕНИЕ TITLE + BODY
# ================================

if not {"category", "title", "body"}.issubset(df.columns):
    print("Ошибка: в файле нет нужных колонок (category, title, body)")
    exit()

print("Формирование текстового поля...")

df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
df = df[["text", "category"]]
df = df[df["text"].str.strip() != ""]

print("После очистки строк осталось:", len(df))

print("\nРаспределение по категориям:")
print(df["category"].value_counts())


# ================================
# СОХРАНЕНИЕ
# ================================

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
df.to_csv(output_path, index=False, encoding="utf-8")

print("\nФайл успешно сохранён:")
print(output_path)
print("Подготовка датасета завершена.")
