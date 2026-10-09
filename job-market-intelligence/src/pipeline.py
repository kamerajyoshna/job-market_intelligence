"""End-to-end: raw dataframe -> cleaned dataframe with extracted skills."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from .data_loader import detect_columns
from .preprocessing import clean_dataframe
from .skill_extractor import SkillExtractor


def process(raw: pd.DataFrame, mapping: Optional[Dict[str, Optional[str]]] = None,
            extractor: Optional[SkillExtractor] = None) -> pd.DataFrame:
    mapping = mapping or detect_columns(raw)
    extractor = extractor or SkillExtractor()
    cleaned = clean_dataframe(raw, mapping)
    cleaned["skills"] = extractor.extract_series(cleaned["text"])
    return cleaned
