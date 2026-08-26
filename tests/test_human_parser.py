from rob_pipeline.human_parser import html_to_text, parse_human_row, parse_support_items


def test_support_parser_splits_multiple_ids_in_one_paragraph():
    source = (
        "<p>1a.1: PY; 1a.2: Y. Quote: &quot;allocation was concealed&quot;</p>"
        "<p>1b.1: NI. Note: No useful information.</p>"
    )
    items = parse_support_items(source, "Domain 1")
    assert [item["question_id"] for item in items] == ["1a.1", "1a.2", "1b.1"]
    assert items[0]["response"] == "Probably Yes"
    assert items[1]["response"] == "Yes"
    assert items[2]["response"] == "No information"
    assert items[0]["canonical_question_id"] == "1.1"
    assert items[2]["canonical_question_id"] is None


def test_human_row_uses_matching_judgement_columns():
    row = {
        "Bias arising from the randomization process (support)": "<p>1.1: Y.</p>",
        "Bias arising from the randomization process (judgement)": "Low risk",
        "Bias due to deviations from intended interventions (support)": "<p>2.1: PY.</p>",
        "Bias due to deviations from intended interventions (judgement)": "Some concerns",
        "Bias due to missing outcome data (support)": "<p>3.1: Y.</p>",
        "Bias due to missing outcome data (judgement)": "Low risk",
        "Bias in measurement of the outcome (support)": "<p>4.1: PN.</p>",
        "Bias in measurement of the outcome (judgement)": "Some concerns",
        "Bias in selection of the reported result (support)": "<p>5.1: Y.</p>",
        "Bias in selection of the reported result (judgement)": "Low risk",
        "Overall bias (support)": "<p>Some concerns in one domain.</p>",
        "Overall bias (judgement)": "Some concerns",
    }
    items = parse_human_row(row)
    conclusions = {item["question_id"]: item["response"] for item in items if "Conclusion" in item["question_id"]}
    assert conclusions["Domain_1_Conclusion"] == "Low risk"
    assert conclusions["Domain_2_Conclusion"] == "Some concerns"
    assert items[-1]["response"] == "Some concerns"
    assert html_to_text(row["Overall bias (support)"]) == "Some concerns in one domain."
