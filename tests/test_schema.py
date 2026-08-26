from rob_pipeline.schema import canonical_topic, normalize_response_label, response_to_number


def test_topic_normalization_handles_variants():
    assert canonical_topic("Distress/PTSD symptoms at\u00a00-1 months")[0] == "PTSD0_1"
    assert canonical_topic("Diagnosis of mental disorders at 1 to 6 months")[0] == "Diagnosis1_6"
    assert canonical_topic("Adverse events at 0-1 months")[0] == "Adverse_event"


def test_response_normalization():
    assert normalize_response_label("PY") == "Probably Yes"
    assert normalize_response_label("No information.") == "No information"
    assert response_to_number("Some concerns") == 2
