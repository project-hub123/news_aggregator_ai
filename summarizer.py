import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_summary(text, num_sentences=3):

    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 40]

    if len(sentences) < 2:
        return "Недостаточно текста для генерации аннотации."

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

    scores = np.array(X.sum(axis=1)).flatten()

    top_indices = scores.argsort()[-num_sentences:]
    top_indices.sort()

    summary = ". ".join([sentences[i] for i in top_indices])

    return summary + "."
