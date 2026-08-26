import csv
import json
from pathlib import Path

import pytest

from hallucination_qe import _load_human_lookup
from hallucination_report import load_rows, summarize_block
from rob_pipeline.io_utils import deduplicate_study_topics, find_response_jsons


def _rob_row(study="Example 2020", analysis="Anxiety symptoms at 0-1 months", judgement="Low risk"):
    row = {"Study": study, "Analysis name": analysis}
    for domain in (
        "Bias arising from the randomization process",
        "Bias due to deviations from intended interventions",
        "Bias due to missing outcome data",
        "Bias in measurement of the outcome",
        "Bias in selection of the reported result",
        "Overall bias",
    ):
        row[f"{domain} (judgement)"] = judgement
        row[f"{domain} (support)"] = "<p>1.1: Y.</p>"
    return row


def test_deduplication_collapses_identical_topic_variants():
    rows = [
        _rob_row(analysis="Depressive symptoms at 0-1 months"),
        _rob_row(analysis="Depressive symptoms at 0-1 months—indicated prevention adults"),
    ]
    records = deduplicate_study_topics(rows)
    assert len(records) == 1
    assert records[0].topic_slug == "Depressive0_1"


def test_deduplication_rejects_conflicting_annotations():
    rows = [_rob_row(), _rob_row(judgement="High risk")]
    with pytest.raises(ValueError, match="Conflicting RoB annotations"):
        deduplicate_study_topics(rows)


def test_hallucination_summary(tmp_path):
    payload = [
        {
            "study": "Example 2020",
            "question_id": "1.1",
            "machine_response": "Yes",
            "human_response": "Yes",
            "topic": "Anxiety symptoms at 0-1 months",
            "topic_slug": "Anxiety0_1",
            "generator_model": "gpt-5",
            "bm25_top_score": 10.0,
            "bm25_mean_topk": 6.0,
            "verdict": {"verdict": "Supported", "quote": "x", "rationale": "x"},
        },
        {
            "study": "Example 2020",
            "question_id": "1.2",
            "machine_response": "Yes",
            "human_response": "No",
            "topic": "Anxiety symptoms at 0-1 months",
            "topic_slug": "Anxiety0_1",
            "generator_model": "gpt-5",
            "bm25_top_score": 4.0,
            "bm25_mean_topk": 2.0,
            "verdict": {"verdict": "Not Found", "quote": "", "rationale": "x"},
        },
    ]
    path = tmp_path / "example.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = load_rows(tmp_path)
    summary = summarize_block(rows)
    assert summary["n_items"] == 2
    assert summary["support_percent"] == 50.0
    assert summary["conservative_hallucination_percent"] == 50.0
    assert summary["strict_hallucination_percent"] == 50.0
    assert summary["mean_bm25_top_score"] == 7.0


def test_find_response_jsons_ignores_generated_reports(tmp_path: Path):
    response = tmp_path / "Anxiety0_1" / "gpt-5" / "Study 2020.json"
    report = tmp_path / "reports" / "summary.json"
    nested_report = tmp_path / "topic" / "tables" / "overall_summary.json"
    for path in (response, report, nested_report):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")

    assert find_response_jsons(tmp_path) == [response]


def test_human_lookup_uses_unambiguous_canonical_response(tmp_path: Path):
    path = tmp_path / "Anxiety0_1" / "Example 2020" / "Example 2020.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "1a.1",
                    "canonical_question_id": "1.1",
                    "response": "Yes",
                },
                {
                    "question_id": "Domain_1_Conclusion",
                    "canonical_question_id": "Domain_1_Conclusion",
                    "response": "Low risk",
                },
            ]
        ),
        encoding="utf-8",
    )

    lookup, source = _load_human_lookup(tmp_path, "Anxiety0_1", "Example 2020")

    assert lookup["1.1"] == "Yes"
    assert lookup["domain_1_conclusion"] == "Low risk"
    assert source == str(path)
