import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_summary(text, max_sentences=1, max_chars=350):
    """
    Генерация краткой аннотации на основе TF-IDF.
    Возвращает действительно сокращённый текст.
    """

    if not text or len(text.strip()) < 50:
        return "Введите более развернутый текст (минимум 2–3 предложения)."

    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 2:
        return "Текст слишком короткий для сокращения."

    # Векторизация предложений
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    # Оценка значимости предложений
    scores = np.array(X.sum(axis=1)).flatten()

    # Сортировка по убыванию значимости
    ranked = sorted(
        [(scores[i], sentences[i]) for i in range(len(sentences))],
        reverse=True
    )

    # Берём только самые важные предложения
    selected_sentences = [ranked[i][1] for i in range(min(max_sentences, len(ranked)))]

    summary = " ".join(selected_sentences)

    # Ограничение по длине
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."

    return summary
