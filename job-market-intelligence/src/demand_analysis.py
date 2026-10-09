"""Skill demand, role/location comparison, and co-occurrence analytics."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def skill_frequency(skill_lists: pd.Series) -> pd.DataFrame:
    """Columns: skill, count, demand_pct (% of postings mentioning the skill)."""
    n = len(skill_lists)
    if n == 0:
        return pd.DataFrame(columns=["skill", "count", "demand_pct"])
    counts = skill_lists.explode().dropna().value_counts()
    df = counts.rename_axis("skill").reset_index(name="count")
    df["demand_pct"] = (df["count"] / n * 100).round(2)
    return df


def skill_matrix(skill_lists: pd.Series) -> pd.DataFrame:
    """Binary posting x skill matrix."""
    mlb = MultiLabelBinarizer()
    m = mlb.fit_transform(skill_lists.tolist()) if len(skill_lists) else np.zeros((0, 0))
    return pd.DataFrame(m, columns=mlb.classes_, index=skill_lists.index)


def skills_by_group(df: pd.DataFrame, group_col: str, top_groups: int = 8, top_skills: int = 12,
                    min_group_size: int = 5) -> pd.DataFrame:
    """Demand % of top skills within each group (rows=skill, cols=group)."""
    sizes = df[group_col].value_counts()
    groups = [g for g in sizes.index if sizes[g] >= min_group_size][:top_groups]
    if not groups:
        return pd.DataFrame()
    cols = {}
    for g in groups:
        f = skill_frequency(df.loc[df[group_col] == g, "skills"]).set_index("skill")["demand_pct"]
        cols[g] = f
    table = pd.DataFrame(cols).fillna(0.0)
    top = table.max(axis=1).sort_values(ascending=False).head(top_skills).index
    return table.loc[top]


def cooccurrence(skill_lists: pd.Series, top_n: int = 15) -> pd.DataFrame:
    """Co-occurrence counts among the top_n most frequent skills."""
    m = skill_matrix(skill_lists)
    if m.shape[1] == 0:
        return pd.DataFrame()
    top = m.sum().sort_values(ascending=False).head(top_n).index
    sub = m[top].values
    return pd.DataFrame(sub.T @ sub, index=top, columns=top)


def example_postings(df: pd.DataFrame, skill: str, n: int = 3) -> pd.DataFrame:
    """A few postings that mention a skill (title + description snippet)."""
    mask = df["skills"].map(lambda s: skill in s)
    out = df.loc[mask, ["title", "location", "description"]].head(n).copy()
    out["description"] = out["description"].str.slice(0, 280)
    return out


def trend_over_time(df: pd.DataFrame, skill: str, freq: str = "M", min_postings: int = 5) -> Optional[pd.DataFrame]:
    """Monthly demand % for a skill; None if dates are missing/insufficient."""
    d = df.dropna(subset=["date"])
    if len(d) < 2 * min_postings:
        return None
    d = d.assign(period=d["date"].dt.to_period(freq).dt.to_timestamp())
    g = d.groupby("period")["skills"].agg(
        postings="size", mentions=lambda s: sum(skill in x for x in s))
    g = g[g["postings"] >= min_postings]
    if len(g) < 2:
        return None
    g["demand_pct"] = (g["mentions"] / g["postings"] * 100).round(2)
    return g.reset_index()
