"""Check that the code gives the numbers reported in the paper.

Needs the data files in data/ (see data/README.md). Run with: pytest
"""
from pathlib import Path

import pytest

from rct import tables
from rct.data import load_study
from rct.metrics import participant_metrics, to_itt

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "reference_standard.csv").exists(),
                                reason="data files not found in data/")


@pytest.fixture(scope="module")
def study():
    return load_study(DATA)


@pytest.fixture(scope="module")
def per_protocol(study):
    return participant_metrics(study)


@pytest.fixture(scope="module")
def itt(study, per_protocol):
    return to_itt(per_protocol, study.n_randomized)


def test_sample_sizes(study, per_protocol, itt):
    assert study.n_randomized == {"Control": 40, "Experimental": 38}
    assert per_protocol["Arm"].value_counts().to_dict() == {"Control": 26, "Experimental": 37}
    assert itt["Arm"].value_counts().to_dict() == {"Control": 40, "Experimental": 38}


def test_reference_standard(study):
    assert len(study.all_ids) == 585
    assert len(study.reference) == 142
    assert len(study.gold) == 100
    t = tables.table2_reference_standard(study).set_index("Characteristic")
    assert t.loc["Orthographic", "n"] == 74 and t.loc["Grammatical", "n"] == 68
    assert t.loc["Punctuation", "n"] == 38
    assert t.loc["3 or more errors", "n"] == 8


def test_table3(itt):
    descriptives, tests = tables.table3_descriptives(itt)
    assert descriptives.loc["Mean", "Control - Total Responses"] == "1.88"
    assert descriptives.loc["Mean", "Experimental - Precision"] == "86.83%"
    tests = tests.set_index("Variable")
    assert tests.loc["Responses", "U"] == 387.0 and tests.loc["Responses", "p"] == "0.0002"
    assert tests.loc["Precision", "U"] == 300.0 and tests.loc["Precision", "p"] == "<0.0001"


def test_table4(itt):
    t = tables.table4_quality(itt).set_index("Metric")
    assert t.loc["Mean Accuracy (SE)", "Control"] == "56.76% (6.68)"
    assert t.loc["Mean Accuracy (SE)", "Experimental"] == "84.77% (2.36)"
    assert t.loc["Mean Recall (SE)", "Experimental"] == "17.81% (1.66)"
    assert t.loc["Mean F1-Score (SE)", "Control"] == "13.65% (3.26)"
    assert t.loc["Mean Kappa (SE)", "Experimental"] == "0.2512 (0.0222)"
    assert list(t["U"]) == [558.0, 344.0, 337.5, 324.0]
    assert t.loc["Mean Accuracy (SE)", "p-value (Holm)"] == "0.0430"


def test_table5(study):
    main, top, exposure, mcnemar = tables.table5_ai_vs_assisted(study)
    row = main.set_index("Condition")
    assert list(row.loc["AI-Only"]) == [0.8496, 0.6579, 0.25, 0.3623, 0.296]
    assert list(row.loc["AI-Assisted (pooled, n=37)"]) == [0.8513, 0.6327, 0.31, 0.4161, 0.3421]
    # a participant with no flagged sentence cannot change the pooled union
    assert list(row.loc["AI-Assisted (pooled, n=38)"]) == list(row.loc["AI-Assisted (pooled, n=37)"])
    top = top.set_index("Condition")
    assert list(top.loc["Educator 1"]) == [0.8513, 0.697, 0.23, 0.3459, 0.2852]
    assert list(top.loc["Educator 5"]) == [0.841, 0.8889, 0.08, 0.1468, 0.122]
    assert list(exposure.iloc[0, 1:]) == [0.8538, 0.835, 0.1552, 0.2618, 0.221]
    assert mcnemar.set_index("Comparison with AI-Only").loc["AI-Assisted (pooled)", "McNemar p"] == "0.0010"


def test_table7(per_protocol, itt):
    t = tables.table7_sensitivity(per_protocol, itt).set_index("Sample")
    assert t.loc["All randomized (40/38)", "Precision"] == "36.52 / 86.83"
    assert t.loc["All, no outliers (37/36)", "Recall"] == "7.64 / 17.52"
    assert t.loc[">=1 record (26/37)", "Recall"] == "14.85 / 18.29"
    assert t.loc[">=1 record, no outliers (23/34)", "F1"] == "18.22 / 28.65"


def test_participation_and_correlation(study, per_protocol, itt):
    p = tables.participation(study, per_protocol).set_index("Arm")
    assert p.loc["Control", "No records"] == 14 and p.loc["Experimental", "No records"] == 1
    assert p.loc["Control", "Median pages reached (with records)"] == 3
    assert p.loc["Experimental", "Median pages reached (with records)"] == 5
    assert p.loc["Control", "Fisher exact p"] == "0.0003"
    rho = tables.spearman(itt).set_index("Arm")["Spearman rho"]
    assert round(rho["Control"], 2) == 0.85 and round(rho["Experimental"], 2) == 0.98
