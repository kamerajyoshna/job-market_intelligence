"""Loading CSVs, auto-detecting columns, and generating a LABELLED synthetic fallback dataset."""
from __future__ import annotations

import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import RANDOM_SEED

COLUMN_SYNONYMS = {
    "title": ["job_title", "title", "position", "role", "designation", "jobtitle", "job_role"],
    "company": ["company", "company_name", "employer", "organization", "org", "hiring_company", "recruiter"],
    "description": ["job_description", "description", "details", "summary", "jobdescription", "job_desc"],
    "skills": ["skills", "required_skills", "skill", "key_skills", "skills_required", "skill_set", "requirements"],
    "location": ["location", "city", "job_location", "region", "country", "work_location"],
    "salary": ["salary", "salary_range", "pay", "compensation", "salary_estimate", "ctc", "salary_text"],
    "date": ["date", "posted_date", "date_posted", "posted", "published", "post_date", "posting_date"],
}


class DataLoadError(Exception):
    """Raised when a dataset cannot be read or is unusable."""


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def load_csv(source) -> pd.DataFrame:
    """Load a CSV from a path or file-like object. Tries utf-8, then latin-1."""
    last_err = None
    for enc in ("utf-8", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
        except pd.errors.EmptyDataError as e:
            raise DataLoadError("The file is empty.") from e
        except Exception as e:  # malformed CSV, missing file, ...
            raise DataLoadError(f"Could not read the CSV: {e}") from e
    else:
        raise DataLoadError(f"Could not decode the file: {last_err}")
    if df.empty or df.shape[1] == 0:
        raise DataLoadError("The CSV contains no rows or no columns.")
    return df


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map canonical fields -> actual column names (None if not found)."""
    normalized = {_norm(c): c for c in df.columns}
    return {field: next((normalized[n] for n in names if n in normalized), None)
            for field, names in COLUMN_SYNONYMS.items()}


# ---------------------------------------------------------------- synthetic data
_ROLES = {
    # role: (core skills, optional skills, base salary used ONLY to fabricate demo salaries)
    "Data Scientist": (["python", "machine learning", "statistics", "sql"], ["pandas", "scikit-learn", "tensorflow", "natural language processing", "data visualization", "deep learning", "aws"], 90000),
    "Data Analyst": (["sql", "excel", "data visualization", "data analysis"], ["power bi", "tableau", "python", "statistics", "communication"], 60000),
    "Machine Learning Engineer": (["python", "machine learning", "deep learning", "mlops"], ["pytorch", "tensorflow", "docker", "kubernetes", "aws", "git"], 105000),
    "Data Engineer": (["sql", "python", "etl", "spark"], ["airflow", "kafka", "aws", "snowflake", "docker", "data modeling"], 95000),
    "DevOps Engineer": (["docker", "kubernetes", "ci/cd", "linux"], ["aws", "terraform", "bash", "azure", "git"], 98000),
    "Full Stack Developer": (["javascript", "react", "node.js", "sql"], ["typescript", "mongodb", "html", "css", "rest api", "git"], 85000),
    "Frontend Developer": (["javascript", "html", "css", "react"], ["typescript", "git", "rest api", "angular"], 75000),
    "Backend Developer": (["java", "spring boot", "sql", "rest api"], ["docker", "aws", "git", "kafka", "mongodb"], 88000),
}
_TITLE_VARIANTS = ["{r}", "Senior {r}", "Junior {r}", "{r} II", "Sr. {r}", "{r} (Remote)"]
_ALIASES = {"machine learning": ["ML", "machine learning"], "natural language processing": ["NLP", "natural language processing"],
            "power bi": ["PowerBI", "Power BI"], "scikit-learn": ["sklearn", "scikit-learn"], "kubernetes": ["k8s", "Kubernetes"]}
_LOCS = [("Bengaluru, India", "INR"), ("Hyderabad, India", "INR"), ("Pune, India", "INR"),
         ("London, UK", "GBP"), ("New York, USA", "USD"), ("Remote", "USD")]
_COMPANIES = ["Apex AI Labs", "CloudScale Systems", "DataNova Analytics", "FinTech Horizons",
              "Nexus Infotech", "Quantum Edge Solutions", "CyberCore Technologies", "Global Systems Inc."]
_FX = {"INR": 30.0, "GBP": 0.8, "USD": 1.0}  # arbitrary scaling for fake salaries only


def generate_synthetic_dataset(n: int = 600, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a SYNTHETIC job dataset. NOT real job-market data - for demos and tests only."""
    rng = np.random.default_rng(seed)
    roles = list(_ROLES)
    rows = []
    for i in range(n):
        role = roles[rng.integers(len(roles))]
        core, opt, base = _ROLES[role]
        skills = [s for s in core if rng.random() < 0.85] + [s for s in opt if rng.random() < 0.4]
        if not skills:
            skills = [core[0]]
        shown = [str(rng.choice(_ALIASES[s])) if s in _ALIASES else s for s in skills]
        rng.shuffle(shown)
        title = str(rng.choice(_TITLE_VARIANTS)).format(r=role)
        company = str(rng.choice(_COMPANIES))
        loc, cur = _LOCS[rng.integers(len(_LOCS))]
        listing = ", ".join(shown[:-1]) + " and " + shown[-1] if len(shown) > 1 else shown[0]
        desc = (f"We are hiring a {title} at {company} in {loc}. You will design, build and maintain solutions "
                f"for our team. Required experience with {listing}. Strong problem solving is expected.")
        salary = ""
        if rng.random() > 0.25:  # ~25% missing
            level = 0.75 if title.startswith("Junior") else 1.3 if title.startswith(("Senior", "Sr.")) else 1.0
            amt = int(base * level * _FX[cur] * rng.normal(1.0, 0.12) // 1000 * 1000)
            salary = f"{cur} {amt:,} per year"
        rows.append({"job_id": i + 1, "job_title": title, "company": company, "job_description": desc,
                     "required_skills": ", ".join(shown) if rng.random() < 0.5 else "",
                     "location": loc, "salary": salary,
                     "data_source": "SYNTHETIC (generated, not real market data)"})
    return pd.DataFrame(rows)
