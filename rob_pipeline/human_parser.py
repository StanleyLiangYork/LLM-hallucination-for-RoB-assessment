"""Deterministic parsing of Cochrane RoB 2 support fields."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any

from .schema import normalize_response_label, response_to_number


class _SupportHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"p", "br", "div", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"p", "div", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts)).replace("\xa0", " ")
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\s*\n\s*", "\n", value)
        value = re.sub(r"\n{2,}", "\n", value)
        return value.strip()


ITEM_MARKER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<id>[1-5](?:[ab])?\.[1-7](?:[ab])?)(?=\s*[:.]|\s+(?:Y|PY|PN|N|NI|NA|DISCR)\b)",
    re.IGNORECASE,
)
CODE_RE = re.compile(
    r"^\s*[:.]*\s*(?P<code>DISCR|Probably\s+Yes|Probably\s+No|No\s+information|Not\s+Applicable|PY|PN|NI|NA|Yes|No|Y|N)\b\s*[:;,.=-]?\s*",
    re.IGNORECASE,
)


def html_to_text(value: str) -> str:
    parser = _SupportHTMLParser()
    parser.feed(value or "")
    parser.close()
    return parser.text()


def canonical_component_id(raw_id: str) -> str | None:
    qid = raw_id.casefold()
    if qid in {
        "1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
        "3.1", "3.2", "3.3", "3.4", "4.1", "4.2", "4.3", "4.4", "4.5",
        "5.1", "5.2", "5.3",
    }:
        return qid
    cluster_map = {"1a.1": "1.1", "1a.2": "1.2", "1a.3": "1.3"}
    if qid in cluster_map:
        return cluster_map[qid]
    component = re.fullmatch(r"([234]\.\d)[ab]", qid)
    return component.group(1) if component else None


def parse_support_items(support_html: str, domain: str) -> list[dict[str, Any]]:
    text = html_to_text(support_html)
    domain_number_match = re.search(r"[1-5]", domain)
    domain_number = domain_number_match.group(0) if domain_number_match else ""
    matches = [
        match for match in ITEM_MARKER_RE.finditer(text)
        if not domain_number or match.group("id").startswith(domain_number)
    ]
    items: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.end() : end].strip()
        code_match = CODE_RE.match(segment)
        raw_code = code_match.group("code") if code_match else ""
        comment = segment[code_match.end() :] if code_match else segment
        comment = re.sub(r"^\s*(?:Note|Comment)\s*:\s*", "", comment, flags=re.IGNORECASE)
        comment = comment.strip(" \n;.-")
        response = normalize_response_label(raw_code)
        if raw_code.casefold() == "discr":
            response = "Discretion"
        raw_id = match.group("id").lower()
        source_text = (match.group(0) + segment).strip()
        candidate = {
            "question_id": raw_id,
            "canonical_question_id": canonical_component_id(raw_id),
            "domain": domain,
            "response": response,
            "response_raw": raw_code or None,
            "numerical": response_to_number(response),
            "comment": comment,
            "source_text": source_text,
            "source_occurrences": [source_text],
        }
        if raw_id not in by_id:
            by_id[raw_id] = candidate
            items.append(candidate)
            continue

        existing = by_id[raw_id]
        existing["source_occurrences"].append(source_text)
        if existing["response"] is None and response is not None:
            existing["response"] = response
            existing["response_raw"] = raw_code or None
            existing["numerical"] = response_to_number(response)
        elif response is not None and existing["response"] != response:
            conflicts = existing.setdefault("conflicting_responses", [])
            if response not in conflicts:
                conflicts.append(response)
        if comment and comment not in existing["comment"]:
            existing["comment"] = " ".join(
                value for value in (existing["comment"], comment) if value
            )
        existing["source_text"] = "\n".join(existing["source_occurrences"])
    return items


DOMAIN_COLUMNS = (
    (
        "Domain 1",
        "Bias arising from the randomization process (support)",
        "Bias arising from the randomization process (judgement)",
        "Domain_1_Conclusion",
    ),
    (
        "Domain 2",
        "Bias due to deviations from intended interventions (support)",
        "Bias due to deviations from intended interventions (judgement)",
        "Domain_2_Conclusion",
    ),
    (
        "Domain 3",
        "Bias due to missing outcome data (support)",
        "Bias due to missing outcome data (judgement)",
        "Domain_3_Conclusion",
    ),
    (
        "Domain 4",
        "Bias in measurement of the outcome (support)",
        "Bias in measurement of the outcome (judgement)",
        "Domain_4_Conclusion",
    ),
    (
        "Domain 5",
        "Bias in selection of the reported result (support)",
        "Bias in selection of the reported result (judgement)",
        "Domain_5_Conclusion",
    ),
)


def parse_human_row(row: dict[str, str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for domain, support_column, judgement_column, conclusion_id in DOMAIN_COLUMNS:
        support = row.get(support_column) or ""
        items.extend(parse_support_items(support, domain))
        judgement = normalize_response_label(row.get(judgement_column))
        items.append(
            {
                "question_id": conclusion_id,
                "canonical_question_id": conclusion_id,
                "domain": domain,
                "response": judgement,
                "response_raw": row.get(judgement_column) or None,
                "numerical": response_to_number(judgement),
                "comment": "",
                "source_text": (row.get(judgement_column) or "").strip(),
            }
        )

    overall_judgement = normalize_response_label(row.get("Overall bias (judgement)"))
    items.append(
        {
            "question_id": "Overall_risk_of_bias",
            "canonical_question_id": "Overall_risk_of_bias",
            "domain": "Overall",
            "response": overall_judgement,
            "response_raw": row.get("Overall bias (judgement)") or None,
            "numerical": response_to_number(overall_judgement),
            "comment": html_to_text(row.get("Overall bias (support)") or ""),
            "source_text": html_to_text(row.get("Overall bias (support)") or ""),
        }
    )
    return items
