"""Transparent, data-driven skill-gap analysis: priority = demand % among postings for the target role."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from .demand_analysis import skill_frequency


def role_skill_demand(df: pd.DataFrame, role: str, role_col: str = "title_family"):
    """Return (demand_df, n_postings) for a role. Raises ValueError if the role has no postings."""
    sub = df[df[role_col] == role]
    if sub.empty:
        raise ValueError(f"No postings found for role '{role}'.")
    return skill_frequency(sub["skills"]), len(sub)


def _tier(pct: float) -> str:
    return "High" if pct >= 50 else "Medium" if pct >= 25 else "Low"


def analyze_gap(user_skills: Iterable[str], demand: pd.DataFrame, role: str, n_postings: int,
                top_n: int = 15, min_demand_pct: float = 5.0) -> dict:
    """Compare user skills with the top-N most demanded skills for a role.

    coverage_pct = demand-weighted share of the top-N skills the user already has.
    """
    user = {s.strip().lower() for s in user_skills if str(s).strip()}
    target = demand[demand["demand_pct"] >= min_demand_pct].head(top_n).copy()
    if target.empty:
        return {"matched": target, "missing": target, "coverage_pct": 0.0, "roadmap": []}
    target["have"] = target["skill"].str.lower().isin(user)
    matched = target[target["have"]].drop(columns="have").reset_index(drop=True)
    missing = target[~target["have"]].drop(columns="have").reset_index(drop=True)
    missing["priority"] = missing["demand_pct"].map(_tier)
    missing["rationale"] = [f"Appears in {int(c)} of {n_postings} {role} postings ({p:.1f}%)."
                            for c, p in zip(missing["count"], missing["demand_pct"])]
    total = target["demand_pct"].sum()
    coverage = float(matched["demand_pct"].sum() / total * 100) if total else 0.0
    roadmap = []
    for i, (tier, label) in enumerate([("High", "Phase 1 - Core skills"), ("Medium", "Phase 2 - Strengthen profile"),
                                       ("Low", "Phase 3 - Differentiators")], 1):
        items = missing[missing["priority"] == tier]
        if not items.empty:
            roadmap.append({"phase": label, "skills": items["skill"].tolist(),
                            "why": items["rationale"].tolist()})
    return {"matched": matched, "missing": missing, "coverage_pct": round(coverage, 1), "roadmap": roadmap}
