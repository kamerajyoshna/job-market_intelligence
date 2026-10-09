"""Cleaning: text normalization, salary parsing, title normalization, data-quality report."""
from __future__ import annotations

import html
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PERIOD_MULTIPLIERS, TITLE_RULES

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(lpa|lakhs?|k|m)?\b")


def clean_text(text) -> str:
    """Lowercase, unescape HTML, strip tags/URLs, collapse whitespace. Non-strings -> ''."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _TAG_RE.sub(" ", t)
    t = re.sub(r"https?://\S+", " ", t)
    return _WS_RE.sub(" ", t.lower()).strip()


def normalize_title(title) -> str:
    """Map a raw job title to a role family; fall back to a tidy title-cased string."""
    t = clean_text(title)
    if not t:
        return "Unknown"
    for pattern, family in TITLE_RULES:
        if re.search(pattern, t):
            return family
    return re.sub(r"[^a-z0-9+#/ ]+", " ", t).strip().title() or "Unknown"


def parse_salary(value) -> Tuple[float, str]:
    """Parse messy salary text into (annual_amount, currency).

    Returns (nan, 'UNSPECIFIED') when invalid. NO currency conversion is performed.
    Ranges use their midpoint. Period words are normalized to annual; a missing
    period is ASSUMED to be annual (documented limitation).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, "UNSPECIFIED"
    s = str(value).strip().lower().replace(",", "")
    if not s:
        return np.nan, "UNSPECIFIED"
    if "₹" in s or re.search(r"\binr\b|\brs\b|lpa|lakh", s):
        cur = "INR"
    elif "$" in s or re.search(r"\busd\b", s):
        cur = "USD"
    elif "£" in s or re.search(r"\bgbp\b", s):
        cur = "GBP"
    elif "€" in s or re.search(r"\beur\b", s):
        cur = "EUR"
    else:
        cur = "UNSPECIFIED"
    period = "year"
    if re.search(r"hour|/hr|\bhr\b", s):
        period = "hour"
    elif "week" in s:
        period = "week"
    elif "month" in s:
        period = "month"
    if re.match(r"^[^\d]*-\s*\d", s):  # leading minus sign -> negative / invalid
        return np.nan, "UNSPECIFIED"
    mult = {"k": 1e3, "m": 1e6, "lpa": 1e5, "lakh": 1e5, "lakhs": 1e5}
    nums = [float(n) * mult.get(u, 1) for n, u in _NUM_RE.findall(s)]
    if not nums:
        return np.nan, "UNSPECIFIED"
    amount = float(np.mean(nums[:2])) * PERIOD_MULTIPLIERS[period]
    if not np.isfinite(amount) or amount <= 0 or amount > 1e9:
        return np.nan, "UNSPECIFIED"
    return amount, cur


def clean_dataframe(raw: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Return a NEW cleaned dataframe (raw is never modified).

    Output columns: title, description, skills_text, location, salary_raw, title_family,
    salary_annual, currency, date, text
    """
    if raw is None or raw.empty:
        raise ValueError("Dataset is empty.")
    if not mapping.get("title"):
        raise ValueError("A job-title column is required but was not found/selected.")
    if not (mapping.get("description") or mapping.get("skills")):
        raise ValueError("At least one of a description column or a skills column is required.")
    missing = [c for c in mapping.values() if c and c not in raw.columns]
    if missing:
        raise ValueError(f"Mapped columns not present in dataset: {missing}")

    def col(field):
        c = mapping.get(field)
        return raw[c] if c else pd.Series([None] * len(raw), index=raw.index)

    out = pd.DataFrame({
        "title": col("title").map(clean_text),
        "company": col("company").map(lambda x: _WS_RE.sub(" ", str(x)).strip().title()
                                      if pd.notna(x) and str(x).strip() else "Unknown"),
        "description": col("description").map(clean_text),
        "skills_text": col("skills").map(clean_text),
        "location": col("location").map(lambda x: _WS_RE.sub(" ", str(x)).strip().title()
                                         if pd.notna(x) and str(x).strip() else "Unknown"),
        "salary_raw": col("salary"),
    })
    out["title_family"] = out["title"].map(normalize_title)
    parsed = out["salary_raw"].map(parse_salary)
    out["salary_annual"] = parsed.map(lambda p: p[0])
    out["currency"] = parsed.map(lambda p: p[1])
    out["date"] = pd.to_datetime(col("date"), errors="coerce") if mapping.get("date") else pd.NaT
    out["text"] = (out["title"] + " " + out["company"].replace("Unknown", "") + " " +
                   out["description"] + " " + out["skills_text"]).str.strip()
    out = out[((out["description"] != "") | (out["skills_text"] != "")) & (out["title"] != "")]
    out = out.drop_duplicates(subset=["title", "company", "description", "skills_text", "location"]).reset_index(drop=True)
    return out


def data_quality_report(raw: pd.DataFrame, cleaned: Optional[pd.DataFrame] = None) -> dict:
    """Summarize dataset quality before/after cleaning."""
    rep = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(raw.shape[1]),
        "raw_duplicate_rows": int(raw.duplicated().sum()),
        "missing_by_column": raw.isna().sum().to_frame("missing").assign(
            missing_pct=lambda d: (d["missing"] / max(len(raw), 1) * 100).round(1)),
    }
    if cleaned is not None:
        n = max(len(cleaned), 1)
        rep.update({
            "clean_rows": int(len(cleaned)),
            "rows_removed": int(len(raw) - len(cleaned)),
            "valid_salary_rows": int(cleaned["salary_annual"].notna().sum()),
            "valid_salary_pct": round(cleaned["salary_annual"].notna().sum() / n * 100, 1),
            "unknown_location_pct": round((cleaned["location"] == "Unknown").sum() / n * 100, 1),
            "has_dates": bool(cleaned["date"].notna().any()),
            "distinct_role_families": int(cleaned["title_family"].nunique()),
        })
    return rep
