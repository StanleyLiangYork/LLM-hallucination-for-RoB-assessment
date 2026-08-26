"""Canonical RoB 2 questions, labels, JSON schemas, and naming helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobQuestion:
    question_id: str
    domain: str
    text: str
    allowed_responses: tuple[str, ...]


SIGNAL_RESPONSES = (
    "Yes",
    "Probably Yes",
    "Probably No",
    "No",
    "No information",
    "Not Applicable",
)
JUDGEMENT_RESPONSES = ("Low risk", "Some concerns", "High risk")


ROB_QUESTIONS: tuple[RobQuestion, ...] = (
    RobQuestion("1.1", "Domain 1", "Was the allocation sequence random?", SIGNAL_RESPONSES[:-1]),
    RobQuestion(
        "1.2",
        "Domain 1",
        "Was the allocation sequence concealed until participants were enrolled and assigned to interventions?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "1.3",
        "Domain 1",
        "Did baseline differences between intervention groups suggest a problem with the randomization process?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "Domain_1_Conclusion",
        "Domain 1",
        "Assess risk of bias arising from the randomization process.",
        JUDGEMENT_RESPONSES,
    ),
    RobQuestion(
        "2.1",
        "Domain 2",
        "Were participants aware of their assigned intervention during the trial?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "2.2",
        "Domain 2",
        "Were carers and people delivering the interventions aware of participants' assigned intervention during the trial?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "2.3",
        "Domain 2",
        "If 2.1 or 2.2 is Yes, Probably Yes, or No information: Were there deviations from the intended intervention that arose because of the trial context?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "2.4",
        "Domain 2",
        "If 2.3 is Yes or Probably Yes: Were these deviations likely to have affected the outcome?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "2.5",
        "Domain 2",
        "If 2.4 is Yes, Probably Yes, or No information: Were these deviations from intended intervention balanced between groups?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "2.6",
        "Domain 2",
        "Was an appropriate analysis used to estimate the effect of assignment to intervention?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "2.7",
        "Domain 2",
        "If 2.6 is No, Probably No, or No information: Was there potential for a substantial impact of failure to analyse participants in the groups to which they were randomized?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "Domain_2_Conclusion",
        "Domain 2",
        "Assess risk of bias due to deviations from intended interventions (effect of assignment to intervention).",
        JUDGEMENT_RESPONSES,
    ),
    RobQuestion(
        "3.1",
        "Domain 3",
        "Were data for this outcome available for all, or nearly all, participants randomized?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "3.2",
        "Domain 3",
        "If 3.1 is No, Probably No, or No information: Is there evidence that the result was not biased by missing outcome data?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "3.3",
        "Domain 3",
        "If 3.2 is No or Probably No: Could missingness in the outcome depend on its true value?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "3.4",
        "Domain 3",
        "If 3.3 is Yes, Probably Yes, or No information: Is it likely that missingness in the outcome depended on its true value?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "Domain_3_Conclusion",
        "Domain 3",
        "Assess risk of bias due to missing outcome data.",
        JUDGEMENT_RESPONSES,
    ),
    RobQuestion(
        "4.1",
        "Domain 4",
        "Was the method of measuring the outcome inappropriate?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "4.2",
        "Domain 4",
        "Could measurement or ascertainment of the outcome have differed between intervention groups?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "4.3",
        "Domain 4",
        "If 4.1 and 4.2 are No, Probably No, or No information: Were outcome assessors aware of the intervention received by study participants?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "4.4",
        "Domain 4",
        "If 4.3 is Yes, Probably Yes, or No information: Could assessment of the outcome have been influenced by knowledge of intervention received?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "4.5",
        "Domain 4",
        "If 4.4 is Yes, Probably Yes, or No information: Is it likely that assessment of the outcome was influenced by knowledge of intervention received?",
        SIGNAL_RESPONSES,
    ),
    RobQuestion(
        "Domain_4_Conclusion",
        "Domain 4",
        "Assess risk of bias in measurement of the outcome.",
        JUDGEMENT_RESPONSES,
    ),
    RobQuestion(
        "5.1",
        "Domain 5",
        "Were the data that produced this result analysed in accordance with a prespecified analysis plan finalized before unblinded outcome data were available?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "5.2",
        "Domain 5",
        "Is the numerical result likely to have been selected, on the basis of the results, from multiple eligible outcome measurements within the outcome domain?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "5.3",
        "Domain 5",
        "Is the numerical result likely to have been selected, on the basis of the results, from multiple eligible analyses of the data?",
        SIGNAL_RESPONSES[:-1],
    ),
    RobQuestion(
        "Domain_5_Conclusion",
        "Domain 5",
        "Assess risk of bias in selection of the reported result.",
        JUDGEMENT_RESPONSES,
    ),
    RobQuestion(
        "Overall_risk_of_bias",
        "Overall",
        "Based on all domains, assess the overall risk of bias.",
        JUDGEMENT_RESPONSES,
    ),
)

QUESTION_BY_ID = {q.question_id.lower(): q for q in ROB_QUESTIONS}

LABEL_TO_NUMBER = {
    "Yes": 5,
    "Probably Yes": 4,
    "Probably No": 3,
    "No": 2,
    "No information": 1,
    "Not Applicable": 0,
    "High risk": 3,
    "Some concerns": 2,
    "Low risk": 1,
}

_LABEL_ALIASES = {
    "y": "Yes",
    "yes": "Yes",
    "py": "Probably Yes",
    "probably yes": "Probably Yes",
    "pn": "Probably No",
    "probably no": "Probably No",
    "n": "No",
    "no": "No",
    "ni": "No information",
    "no info": "No information",
    "no information": "No information",
    "na": "Not Applicable",
    "n/a": "Not Applicable",
    "not applicable": "Not Applicable",
    "low": "Low risk",
    "low risk": "Low risk",
    "some concern": "Some concerns",
    "some concerns": "Some concerns",
    "high": "High risk",
    "high risk": "High risk",
}


def normalize_response_label(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().strip(".;:")
    return _LABEL_ALIASES.get(text.casefold())


def response_to_number(value: Any) -> int | None:
    label = normalize_response_label(value)
    return LABEL_TO_NUMBER.get(label) if label else None


TOPIC_DISPLAY = {
    "Adverse_event": "Adverse events",
    "Anxiety0_1": "Anxiety symptoms at 0-1 months",
    "Anxiety1_6": "Anxiety symptoms at 1-6 months",
    "Anxiety7_24": "Anxiety symptoms at 7-24 months",
    "Depressive0_1": "Depressive symptoms at 0-1 months",
    "Depressive1_6": "Depressive symptoms at 1-6 months",
    "Depressive7_24": "Depressive symptoms at 7-24 months",
    "Diagnosis0_1": "Diagnosis of mental disorders at 0-1 months",
    "Diagnosis1_6": "Diagnosis of mental disorders at 1-6 months",
    "Diagnosis7_24": "Diagnosis of mental disorders at 7-24 months",
    "Psycho0_1": "Psychological functioning and impairment at 0-1 months",
    "Psycho1_6": "Psychological functioning and impairment at 1-6 months",
    "Psycho7_24": "Psychological functioning and impairment at 7-24 months",
    "PTSD0_1": "Distress/PTSD symptoms at 0-1 months",
    "PTSD1_6": "Distress/PTSD symptoms at 1-6 months",
    "PTSD7_24": "Distress/PTSD symptoms at 7-24 months",
    "QOL0_1": "Quality of life at 0-1 months",
    "QOL1_6": "Quality of life at 1-6 months",
    "QOL7_24": "Quality of life at 7-24 months",
    "Social0_1": "Social outcomes at 0-1 months",
    "Social1_6": "Social outcomes at 1-6 months",
    "Social7_24": "Social outcomes at 7-24 months",
}


def _ascii_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def canonical_topic(analysis_name: str) -> tuple[str, str]:
    """Return the established folder slug and display label for a CSV analysis."""
    text = _ascii_text(analysis_name).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    text = text.replace("1 to 6", "1-6")

    if "adverse event" in text:
        slug = "Adverse_event"
    elif "anxiety" in text:
        slug = "Anxiety"
    elif "depressive" in text:
        slug = "Depressive"
    elif "diagnosis" in text:
        slug = "Diagnosis"
    elif "distress" in text or "ptsd" in text or "post-traumatic" in text:
        slug = "PTSD"
    elif "psychological functioning" in text:
        slug = "Psycho"
    elif "quality of life" in text:
        slug = "QOL"
    elif "social outcome" in text:
        slug = "Social"
    else:
        raise ValueError(f"Unrecognized analysis topic: {analysis_name!r}")

    if slug != "Adverse_event":
        if re.search(r"0\s*-\s*1\s*months?", text):
            slug += "0_1"
        elif re.search(r"1\s*-\s*6\s*months?", text):
            slug += "1_6"
        elif re.search(r"7\s*-\s*24\s*months?", text):
            slug += "7_24"
        else:
            raise ValueError(f"Unrecognized analysis time window: {analysis_name!r}")

    return slug, TOPIC_DISPLAY[slug]


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("._-")
    if not slug:
        raise ValueError("Model name cannot be empty")
    return slug


ROB_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "minItems": len(ROB_QUESTIONS),
            "maxItems": len(ROB_QUESTIONS),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "question_id",
                    "response",
                    "numerical",
                    "comment",
                    "reasoning",
                ],
                "properties": {
                    "question_id": {"type": "string"},
                    "response": {"type": "string"},
                    "numerical": {"type": "integer", "minimum": 0, "maximum": 5},
                    "comment": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
            },
        }
    },
}


VERDICT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "quote", "rationale"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["Supported", "Contradicted", "Not Found", "Out of Scope"],
        },
        "quote": {"type": "string"},
        "rationale": {"type": "string"},
    },
}
