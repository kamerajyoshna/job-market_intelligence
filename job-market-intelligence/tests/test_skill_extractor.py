import pandas as pd

from src.skill_extractor import SkillExtractor

ex = SkillExtractor()


def test_multiword_and_alias_and_case():
    found = ex.extract("Experience in NLP, ML and Natural Language Processing; Power-BI preferred")
    assert {"natural language processing", "machine learning", "power bi"} <= set(found)


def test_java_is_not_javascript():
    assert ex.extract("We use JavaScript daily") == ["javascript"]
    assert "java" not in ex.extract("We use JavaScript daily")


def test_special_character_skills():
    found = ex.extract("C++, C#, Node.js and CI/CD pipelines")
    assert {"c++", "c#", "node.js", "ci/cd"} <= set(found)


def test_no_false_positive_for_substrings():
    assert ex.extract("Experience with sqlalchemy and goal-setting") == []


def test_empty_and_non_string():
    assert ex.extract("") == [] and ex.extract(None) == []


def test_add_skill_is_configurable():
    e = SkillExtractor({"python": []})
    assert e.extract("rust developer") == []
    e.add_skill("rust", ["rustlang"])
    assert e.extract("Rustlang developer") == ["rust"]


def test_canonicalize_user_input():
    assert ex.canonicalize(["PyTorch", "sklearn", "unknownthing"]) == ["pytorch", "scikit-learn"]


def test_extract_series():
    s = ex.extract_series(pd.Series(["python and sql", "nothing"]))
    assert s.tolist() == [["python", "sql"], []]
