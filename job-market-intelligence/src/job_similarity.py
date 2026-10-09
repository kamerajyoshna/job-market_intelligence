"""TF-IDF + cosine similarity between a profile/resume text and job postings.

Similarity measures TEXT OVERLAP only. It is NOT a hiring probability.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobMatcher:
    def __init__(self, texts: pd.Series, meta: pd.DataFrame):
        if len(texts) == 0:
            raise ValueError("Cannot build a matcher from an empty dataset.")
        self.meta = meta.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            self.matrix = self.vectorizer.fit_transform(texts.fillna("").tolist())
        except ValueError as e:  # empty vocabulary
            raise ValueError("Job texts contain no usable words.") from e

    def rank(self, query: str, top_k: int = 10) -> pd.DataFrame:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query text is empty.")
        sims = cosine_similarity(self.vectorizer.transform([query.lower()]), self.matrix).ravel()
        order = sims.argsort()[::-1][:top_k]
        out = self.meta.iloc[order].copy()
        out["similarity"] = sims[order].round(4)
        return out.reset_index(drop=True)
