"""Dictionary + phrase-matching skill extractor.

Limitations (also in docs/methodology.md): cannot find skills missing from the dictionary,
cannot tell "required" from "nice to have" or negations ("no Java needed"), and short
ambiguous tokens (e.g. "go", "r") may produce false positives/negatives.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .config import DEFAULT_SKILLS
from .preprocessing import clean_text


class SkillExtractor:
    def __init__(self, skill_dict: Optional[Dict[str, List[str]]] = None):
        self.skill_dict: Dict[str, List[str]] = {k.lower(): [a.lower() for a in v]
                                                  for k, v in (skill_dict or DEFAULT_SKILLS).items()}
        self._compile()

    def _compile(self) -> None:
        self._patterns = {}
        self._alias_to_skill = {}
        for skill, aliases in self.skill_dict.items():
            terms = sorted({skill, *aliases}, key=len, reverse=True)  # longest first
            for t in terms:
                self._alias_to_skill[t] = skill
            alt = "|".join(re.escape(t) for t in terms)
            # token boundaries that treat + # . / - as part of a skill name (c++, node.js, ci/cd)
            self._patterns[skill] = re.compile(rf"(?<![a-z0-9+#]){alt}(?![a-z0-9+#])")

    def add_skill(self, name: str, aliases: Iterable[str] = ()) -> None:
        self.skill_dict[name.lower()] = [a.lower() for a in aliases]
        self._compile()

    @property
    def skills(self) -> List[str]:
        return sorted(self.skill_dict)

    def extract(self, text) -> List[str]:
        """Return sorted canonical skills found in text (case-insensitive)."""
        t = clean_text(text)
        if not t:
            return []
        return sorted(s for s, p in self._patterns.items() if p.search(t))

    def extract_series(self, texts: pd.Series) -> pd.Series:
        return texts.map(self.extract)

    def canonicalize(self, items: Iterable[str]) -> List[str]:
        """Map user-entered skills/aliases to canonical names; unknown items are dropped."""
        found = set()
        for it in items:
            key = str(it).strip().lower()
            if key in self._alias_to_skill:
                found.add(self._alias_to_skill[key])
            else:
                found.update(self.extract(key))
        return sorted(found)
