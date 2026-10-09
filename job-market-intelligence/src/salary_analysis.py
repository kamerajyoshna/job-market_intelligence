"""Salary analytics. Only compares salaries inside ONE currency - no FX conversion is attempted."""
from __future__ import annotations

import pandas as pd


def salary_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Valid-salary row counts per currency."""
    v = df.dropna(subset=["salary_annual"])
    if v.empty:
        return pd.DataFrame(columns=["currency", "postings"])
    return v["currency"].value_counts().rename_axis("currency").reset_index(name="postings")


def salary_by_group(df: pd.DataFrame, group_col: str, currency: str, min_n: int = 3) -> pd.DataFrame:
    """Annual salary stats by group within a single currency."""
    v = df[(df["currency"] == currency)].dropna(subset=["salary_annual"])
    if v.empty:
        return pd.DataFrame(columns=[group_col, "n", "median", "mean", "p25", "p75"])
    g = v.groupby(group_col)["salary_annual"]
    out = pd.DataFrame({"n": g.size(), "median": g.median(), "mean": g.mean(),
                        "p25": g.quantile(0.25), "p75": g.quantile(0.75)})
    out = out[out["n"] >= min_n].sort_values("median", ascending=False).round(0)
    return out.reset_index()


def skill_salary_association(df: pd.DataFrame, currency: str, min_n: int = 5) -> pd.DataFrame:
    """Median salary with vs without each skill (ASSOCIATION, not causation; confounded by role)."""
    v = df[df["currency"] == currency].dropna(subset=["salary_annual"])
    rows = []
    if len(v) >= 2 * min_n:
        for skill in sorted({s for lst in v["skills"] for s in lst}):
            has = v["skills"].map(lambda l: skill in l)
            if has.sum() >= min_n and (~has).sum() >= min_n:
                w, wo = v.loc[has, "salary_annual"].median(), v.loc[~has, "salary_annual"].median()
                rows.append({"skill": skill, "n_with": int(has.sum()), "median_with": w,
                             "median_without": wo, "difference": w - wo})
    cols = ["skill", "n_with", "median_with", "median_without", "difference"]
    return pd.DataFrame(rows, columns=cols).sort_values("difference", ascending=False).reset_index(drop=True)
