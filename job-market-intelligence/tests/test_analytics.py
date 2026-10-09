import numpy as np
import pandas as pd
import pytest

from src.clustering import cluster_jobs
from src.data_loader import generate_synthetic_dataset
from src.demand_analysis import cooccurrence, skill_frequency, skills_by_group, trend_over_time
from src.job_similarity import JobMatcher
from src.pipeline import process
from src.salary_analysis import salary_by_group, salary_coverage, skill_salary_association
from src.skill_gap import analyze_gap, role_skill_demand


@pytest.fixture(scope="module")
def df():
    return process(generate_synthetic_dataset(400, seed=7))


def test_skill_frequency_known_values():
    s = pd.Series([["python", "sql"], ["python"], [], ["sql", "python"]])
    f = skill_frequency(s).set_index("skill")
    assert f.loc["python", "count"] == 3 and f.loc["python", "demand_pct"] == 75.0
    assert f.loc["sql", "demand_pct"] == 50.0


def test_skill_frequency_empty():
    assert skill_frequency(pd.Series([], dtype=object)).empty


def test_group_and_cooccurrence(df):
    t = skills_by_group(df, "title_family")
    assert not t.empty and (t.values <= 100).all()
    c = cooccurrence(df["skills"], 10)
    assert c.shape == (10, 10) and (c.values == c.values.T).all()


def test_trend_none_without_dates(df):
    assert trend_over_time(df, "python") is None


def test_salary_never_mixes_currencies(df):
    cov = salary_coverage(df)
    assert set(cov["currency"]) <= {"INR", "USD", "GBP"}
    g = salary_by_group(df, "title_family", "INR")
    assert (g["n"] >= 3).all() and g["median"].gt(0).all()
    assert salary_by_group(df, "title_family", "EUR").empty
    assert isinstance(skill_salary_association(df, "INR"), pd.DataFrame)


def test_skill_gap_calculation():
    demand = pd.DataFrame({"skill": ["python", "sql", "docker", "aws"], "count": [90, 80, 40, 10],
                           "demand_pct": [90.0, 80.0, 40.0, 10.0]})
    r = analyze_gap(["Python"], demand, "Data Scientist", 100)
    assert r["matched"]["skill"].tolist() == ["python"]
    assert r["missing"]["skill"].tolist() == ["sql", "docker", "aws"]
    assert r["missing"]["priority"].tolist() == ["High", "Medium", "Low"]
    assert r["coverage_pct"] == pytest.approx(90 / 220 * 100, abs=0.1)
    assert "80 of 100" in r["missing"].loc[0, "rationale"]
    assert [p["phase"][:7] for p in r["roadmap"]] == ["Phase 1", "Phase 2", "Phase 3"]


def test_skill_gap_full_match_and_unknown_role(df):
    demand, n = role_skill_demand(df, "Data Scientist")
    full = analyze_gap(demand["skill"].tolist(), demand, "Data Scientist", n)
    assert full["missing"].empty and full["coverage_pct"] == 100.0
    with pytest.raises(ValueError):
        role_skill_demand(df, "Astronaut")


def test_similarity_ranks_relevant_first(df):
    m = JobMatcher(df["text"], df[["title_family", "title"]])
    top = m.rank("python machine learning statistics scikit-learn data scientist", top_k=5)
    assert len(top) == 5 and top["similarity"].is_monotonic_decreasing
    assert top["title_family"].iloc[0] in {"Data Scientist", "Machine Learning Engineer"}
    assert top["similarity"].between(0, 1).all()


def test_similarity_errors():
    with pytest.raises(ValueError):
        JobMatcher(pd.Series([], dtype=str), pd.DataFrame())
    m = JobMatcher(pd.Series(["python sql", "react css"]), pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(ValueError):
        m.rank("   ")
    assert (m.rank("zzzzqq")["similarity"] == 0).all()


def test_clustering_valid_and_reproducible(df):
    a, b = cluster_jobs(df["text"]), cluster_jobs(df["text"])
    assert not a.skipped and a.k == b.k and -1 <= a.silhouette <= 1
    assert (a.labels == b.labels).all() and len(a.top_terms) == a.k and a.coords.shape == (len(df), 2)


def test_clustering_skips_small_data():
    r = cluster_jobs(pd.Series(["a b", "c d"]))
    assert r.skipped and "at least" in r.reason
