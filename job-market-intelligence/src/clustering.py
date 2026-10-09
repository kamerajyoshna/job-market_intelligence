"""K-Means clustering of job texts on TF-IDF vectors (unsupervised; no accuracy is reported)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from .config import RANDOM_SEED


@dataclass
class ClusterResult:
    skipped: bool = False
    reason: str = ""
    k: int = 0
    silhouette: Optional[float] = None
    silhouette_by_k: dict = field(default_factory=dict)
    labels: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    top_terms: List[List[str]] = field(default_factory=list)


def cluster_jobs(texts: pd.Series, k: Optional[int] = None, k_range=(2, 8), seed: int = RANDOM_SEED,
                 min_docs: int = 30) -> ClusterResult:
    texts = texts.fillna("")
    if len(texts) < min_docs:
        return ClusterResult(skipped=True, reason=f"Need at least {min_docs} postings; found {len(texts)}.")
    vec = TfidfVectorizer(stop_words="english", max_features=2000, min_df=2, sublinear_tf=True)
    try:
        X = vec.fit_transform(texts.tolist())
    except ValueError:
        return ClusterResult(skipped=True, reason="Texts have too little vocabulary to cluster.")
    if X.shape[1] < 2:
        return ClusterResult(skipped=True, reason="Fewer than 2 distinct terms after filtering.")

    sil_by_k = {}
    candidates = [k] if k else range(k_range[0], min(k_range[1], len(texts) - 1) + 1)
    for kk in candidates:
        labels = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(X)
        if 2 <= len(set(labels)) <= len(texts) - 1:  # silhouette is only defined in this range
            sil_by_k[kk] = float(silhouette_score(X, labels))
    if not sil_by_k:
        return ClusterResult(skipped=True, reason="Silhouette score could not be computed.")
    best_k = k if k else max(sil_by_k, key=sil_by_k.get)
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)
    terms = np.array(vec.get_feature_names_out())
    top_terms = [terms[c.argsort()[::-1][:8]].tolist() for c in km.cluster_centers_]
    n_comp = 2 if X.shape[1] > 2 else 1
    coords = TruncatedSVD(n_components=max(n_comp, 1), random_state=seed).fit_transform(X)
    if coords.shape[1] == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])
    return ClusterResult(k=best_k, silhouette=sil_by_k.get(best_k), silhouette_by_k=sil_by_k,
                         labels=km.labels_, coords=coords, top_terms=top_terms)
