import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoadError, detect_columns, generate_synthetic_dataset, load_csv
from src.preprocessing import clean_dataframe, clean_text, data_quality_report, normalize_title, parse_salary


def test_clean_text_handles_html_case_and_non_strings():
    assert clean_text("<p>Hello&amp;  WORLD</p> http://x.com") == "hello& world"
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


@pytest.mark.parametrize("raw,amount,cur", [
    ("$85k per year", 85000, "USD"),
    ("INR 900,000 per year", 900000, "INR"),
    ("£3000 per month", 36000, "GBP"),
    ("$50 per hour", 104000, "USD"),
    ("$60,000 - $80,000", 70000, "USD"),
    ("12 LPA", 1200000, "INR"),
])
def test_parse_salary_valid(raw, amount, cur):
    a, c = parse_salary(raw)
    assert a == pytest.approx(amount) and c == cur


@pytest.mark.parametrize("bad", [None, np.nan, "", "negotiable", "-5", "0", "competitive"])
def test_parse_salary_invalid(bad):
    a, _ = parse_salary(bad)
    assert np.isnan(a)


def test_normalize_title_families():
    assert normalize_title("Sr. Data Scientist II") == "Data Scientist"
    assert normalize_title("ML Engineer") == "Machine Learning Engineer"
    assert normalize_title("Full-Stack Web Dev") == "Full-Stack Developer"
    assert normalize_title("") == "Unknown"


def test_detect_columns():
    df = pd.DataFrame(columns=["Job Title", "Employer", "Job Description", "Salary Range", "City"])
    m = detect_columns(df)
    assert m["title"] == "Job Title" and m["company"] == "Employer"
    assert m["description"] == "Job Description"
    assert m["salary"] == "Salary Range" and m["location"] == "City" and m["skills"] is None


def test_clean_dataframe_removes_duplicates_and_empty_rows_and_keeps_raw():
    raw = pd.DataFrame({"t": ["Data Scientist"] * 2 + ["Analyst", ""],
                        "c": ["Tech Corp", "Tech Corp", "Data Co", "Old Corp"],
                        "d": ["python sql"] * 2 + [None, "x"],
                        "s": ["$50k", "$50k", "abc", "1"]})
    before = raw.copy()
    out = clean_dataframe(raw, {"title": "t", "company": "c", "description": "d", "salary": "s"})
    assert len(out) == 1                       # dup removed, empty description row removed, empty title removed
    pd.testing.assert_frame_equal(raw, before)  # raw untouched
    assert out.loc[0, "salary_annual"] == 50000
    assert out.loc[0, "company"] == "Tech Corp"


def test_clean_dataframe_errors():
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame(), {"title": "t"})
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"a": [1]}), {"title": None, "description": "a"})
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"a": [1]}), {"title": "a"})
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"a": [1]}), {"title": "zzz", "description": "a"})


def test_load_csv_errors(tmp_path):
    empty = tmp_path / "e.csv"; empty.write_text("")
    with pytest.raises(DataLoadError):
        load_csv(str(empty))
    with pytest.raises(DataLoadError):
        load_csv(str(tmp_path / "missing.csv"))


def test_synthetic_is_labelled_and_reproducible():
    a, b = generate_synthetic_dataset(50, seed=1), generate_synthetic_dataset(50, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert a["data_source"].str.contains("SYNTHETIC").all()


def test_quality_report():
    df = generate_synthetic_dataset(60)
    clean = clean_dataframe(df, detect_columns(df))
    rep = data_quality_report(df, clean)
    assert rep["raw_rows"] == 60 and 0 <= rep["valid_salary_pct"] <= 100
