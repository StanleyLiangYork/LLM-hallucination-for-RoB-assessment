#!/usr/bin/env python3
"""Generate structured RoB 2 assessments from trial-report PDFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from rob_pipeline.io_utils import (
    StudyTopicRecord,
    atomic_write_json,
    context_limit_for_model,
    deduplicate_study_topics,
    find_pdf_for_study,
    load_api_key,
    load_json_items,
    read_csv_rows,
    read_pdf_text,
    safe_component,
    token_count,
    truncate_to_token_budget,
    validate_csv_columns,
)
from rob_pipeline.openai_json import OpenAIJsonClient
from rob_pipeline.schema import (
    QUESTION_BY_ID,
    ROB_QUESTIONS,
    ROB_RESPONSE_JSON_SCHEMA,
    canonical_topic,
    model_slug,
    normalize_response_label,
    response_to_number,
)


SYSTEM_PROMPT = """You are an exacting medical-literature reviewer applying the Cochrane RoB 2 framework.
Base every answer exclusively on the supplied trial report and the stated outcome topic. Do not use outside knowledge or fill reporting gaps with assumptions. Use "No information" when the report does not provide enough evidence. Use "Not Applicable" only when a conditional signaling question is not reached. Return one concise evidence-based comment and one brief rationale for every item. The rationale is a summary of the evidentiary basis, not hidden chain-of-thought. Return valid JSON only."""


def _question_block() -> str:
    sections: list[str] = []
    for question in ROB_QUESTIONS:
        allowed = " / ".join(question.allowed_responses)
        sections.append(
            f"{question.question_id} [{question.domain}]: {question.text}\n"
            f"Allowed response: {allowed}"
        )
    return "\n\n".join(sections)


def build_user_prompt(topic: str, paper_text: str) -> str:
    return f"""Outcome topic being assessed:
{topic}

Trial report:
<trial_report>
{paper_text}
</trial_report>

RoB 2 questions:
{_question_block()}

Return exactly one item for each question in the order shown. Use this top-level shape:
{{"items": [{{"question_id": "...", "response": "...", "numerical": 0, "comment": "...", "reasoning": "..."}}]}}

Numerical mappings:
Yes=5; Probably Yes=4; Probably No=3; No=2; No information=1; Not Applicable=0.
High risk=3; Some concerns=2; Low risk=1.
"""


def validate_assessment(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Response must contain an 'items' array")

    by_id: dict[str, dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("Every assessment item must be an object")
        raw_id = str(raw.get("question_id", "")).strip()
        canonical = QUESTION_BY_ID.get(raw_id.casefold())
        if canonical is None:
            raise ValueError(f"Unexpected question_id: {raw_id!r}")
        if canonical.question_id in by_id:
            raise ValueError(f"Duplicate question_id: {canonical.question_id}")

        response = normalize_response_label(raw.get("response"))
        if response is None or response not in canonical.allowed_responses:
            raise ValueError(
                f"Invalid response for {canonical.question_id}: {raw.get('response')!r}"
            )
        numerical = response_to_number(response)
        supplied_number = raw.get("numerical")
        try:
            supplied_number = int(supplied_number)
        except (TypeError, ValueError):
            supplied_number = None
        if supplied_number is not None and supplied_number != numerical:
            raise ValueError(
                f"Incorrect numerical mapping for {canonical.question_id}: "
                f"{response!r} should be {numerical}, got {supplied_number}"
            )

        by_id[canonical.question_id] = {
            "question_id": canonical.question_id,
            "question_text": canonical.text,
            "domain": canonical.domain,
            "response": response,
            "numerical": numerical,
            "comment": str(raw.get("comment") or "").strip(),
            "reasoning": str(raw.get("reasoning") or "").strip(),
        }

    missing = [q.question_id for q in ROB_QUESTIONS if q.question_id not in by_id]
    if missing:
        raise ValueError("Response omitted questions: " + ", ".join(missing))
    return [by_id[q.question_id] for q in ROB_QUESTIONS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_paths(output_root: Path, topic_slug: str, model: str, study: str) -> tuple[Path, Path]:
    folder = output_root / topic_slug / model_slug(model) / safe_component(study)
    return folder / f"{safe_component(study)}.json", folder / "metadata.json"


def _existing_complete(path: Path) -> bool:
    try:
        return len(load_json_items(path)) == len(ROB_QUESTIONS)
    except Exception:
        return False


def _single_record(args: argparse.Namespace) -> tuple[StudyTopicRecord, Path]:
    if not args.topic:
        raise ValueError("--topic is required with --pdf")
    topic_slug, topic = canonical_topic(args.topic)
    pdf_path = Path(args.pdf)
    study = args.study or pdf_path.stem.replace("_", " ")
    record = StudyTopicRecord(
        study=study,
        topic_slug=topic_slug,
        topic=topic,
        analysis_names=(args.topic,),
        csv_row_numbers=(),
        row={},
    )
    return record, pdf_path


def _batch_records(args: argparse.Namespace) -> list[tuple[StudyTopicRecord, Path]]:
    rows, fieldnames = read_csv_rows(args.csv)
    validate_csv_columns(fieldnames, ("Study", "Analysis name"))
    records = deduplicate_study_topics(rows)
    selected: list[tuple[StudyTopicRecord, Path]] = []
    for record in records:
        if args.topic_slug and record.topic_slug != args.topic_slug:
            continue
        if args.study_filter and args.study_filter.casefold() not in record.study.casefold():
            continue
        selected.append((record, find_pdf_for_study(args.pdf_root, record.study)))
    return selected[: args.limit] if args.limit else selected


def process_one(
    record: StudyTopicRecord,
    pdf_path: Path,
    args: argparse.Namespace,
    api_client: OpenAIJsonClient,
) -> dict[str, Any]:
    output_path, metadata_path = _output_paths(
        Path(args.output), record.topic_slug, args.model, record.study
    )
    if output_path.exists() and _existing_complete(output_path) and not args.overwrite:
        return {"status": "skipped", "study": record.study, "topic": record.topic_slug, "output": str(output_path)}

    paper_text, page_count = read_pdf_text(pdf_path)
    fixed_prompt = build_user_prompt(record.topic, "")
    context_limit = context_limit_for_model(args.model, args.context_limit)
    fixed_tokens = token_count(SYSTEM_PROMPT + fixed_prompt, args.model)
    budget = context_limit - args.output_reserve_tokens - fixed_tokens
    bounded_text, input_tokens, truncated = truncate_to_token_budget(
        paper_text, args.model, budget
    )
    if not bounded_text.strip():
        raise ValueError(f"No extractable PDF text for {pdf_path}")

    result = api_client.complete_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(record.topic, bounded_text),
        schema_name="rob2_assessment",
        schema=ROB_RESPONSE_JSON_SCHEMA,
    )
    items = validate_assessment(result.data)
    for item in items:
        item.update(
            {
                "study": record.study,
                "topic": record.topic,
                "topic_slug": record.topic_slug,
            }
        )
    atomic_write_json(output_path, items)
    atomic_write_json(
        metadata_path,
        {
            "study": record.study,
            "topic": record.topic,
            "topic_slug": record.topic_slug,
            "analysis_names": list(record.analysis_names),
            "csv_row_numbers": list(record.csv_row_numbers),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "pdf_file": str(pdf_path.resolve()),
            "pdf_sha256": _sha256(pdf_path),
            "pdf_pages": page_count,
            "extracted_characters": len(paper_text),
            "estimated_input_tokens": input_tokens,
            "input_truncated": truncated,
            "context_limit": context_limit,
            "output_mode": result.output_mode,
            "openai_response_id": result.response_id,
            "finish_reason": result.finish_reason,
        },
    )
    return {"status": "completed", "study": record.study, "topic": record.topic_slug, "output": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assess RoB 2 from one PDF or a CSV-indexed PDF collection using a selected OpenAI model."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", help="Single trial-report PDF")
    source.add_argument("--csv", help="CSV containing Study and Analysis name columns")
    parser.add_argument("--pdf-root", "--pdf_root", dest="pdf_root", help="Root containing <Study>/<Author_Year>.pdf; required with --csv")
    parser.add_argument("--topic", help="Outcome topic for --pdf mode")
    parser.add_argument("--study", help="Study ID override for --pdf mode")
    parser.add_argument("--output", required=True, help="Output root")
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="OpenAI model ID, for example gpt-5, o3-mini, gpt-4o, or gpt-3.5-turbo",
    )
    parser.add_argument("--api-key-file", "--api_key", dest="api_key_file", help="Text file containing the API key; otherwise use OPENAI_API_KEY")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="Reasoning effort for models that support it",
    )
    parser.add_argument("--context-limit", type=int, help="Override model context-window size")
    parser.add_argument("--output-reserve-tokens", type=int, default=6000)
    parser.add_argument("--topic-slug", help="Batch filter such as Anxiety0_1")
    parser.add_argument("--study-filter", help="Batch substring filter")
    parser.add_argument("--limit", type=int, help="Process at most this many study-topic pairs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate discovery and print planned outputs without API calls")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.csv and not args.pdf_root:
        raise ValueError("--pdf-root is required with --csv")
    jobs = [_single_record(args)] if args.pdf else _batch_records(args)
    if not jobs:
        print("No matching study-topic pairs.")
        return 0

    if args.dry_run:
        for record, pdf_path in jobs:
            output_path, _ = _output_paths(Path(args.output), record.topic_slug, args.model, record.study)
            print(f"{record.topic_slug}\t{record.study}\t{pdf_path}\t{output_path}")
        print(f"Validated {len(jobs)} planned assessment(s).")
        return 0

    api_client = OpenAIJsonClient(
        api_key=load_api_key(args.api_key_file),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    results: list[dict[str, Any]] = []
    failures = 0
    for index, (record, pdf_path) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {record.topic_slug} / {record.study}")
        try:
            result = process_one(record, pdf_path, args, api_client)
        except Exception as exc:
            failures += 1
            result = {
                "status": "failed",
                "study": record.study,
                "topic": record.topic_slug,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  ERROR: {result['error']}", file=sys.stderr)
        else:
            print(f"  {result['status']}: {result['output']}")
        results.append(result)

    manifest = Path(args.output) / f"run_manifest_{model_slug(args.model)}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["status", "topic", "study", "output", "error"]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Manifest: {manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
