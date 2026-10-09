"""Headless automated tests for the Streamlit UI using AppTest."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_starts_and_loads_demo():
    at = AppTest.from_file(APP_PATH, default_timeout=10).run()
    assert not at.exception

    # Trigger synthetic demo dataset load
    at.button[0].click().run()
    assert not at.exception
    assert "df" in at.session_state
    assert len(at.session_state["df"]) > 0


def test_all_pages_render_without_exceptions():
    at = AppTest.from_file(APP_PATH, default_timeout=10).run()
    # Load demo data
    at.button[0].click().run()
    assert not at.exception

    pages = [
        "Dashboard Overview",
        "Dataset & Data Quality",
        "Job Market Insights",
        "Skill Demand Analytics",
        "Salary Intelligence",
        "Skill Gap Analyzer",
        "Job Description Similarity",
        "ML Text Clustering",
        "Methodology & Viva Guide",
    ]

    for p in pages:
        at.sidebar.radio[0].set_value(p).run()
        errs = [e.value for e in at.exception]
        assert len(errs) == 0, f"Page {p} raised exceptions: {errs}"
