#!/usr/bin/env python3
"""Retrieve evidence with BM25 and judge support for each RoB 2 claim."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from rob_pipeline.io_utils import (
    atomic_write_json,
    find_pdf_for_study,
    find_response_jsons,
    load_api_key,
    load_json_items,
)
from rob_pipeline.openai_json import OpenAIJsonClient
from rob_pipeline.retrieval import BM25Retriever, chunk_pdf, normalize_whitespace
from rob_pipeline.schema import QUESTION_BY_ID, VERDICT_JSON_SCHEMA


SYSTEM_PROMPT = """You verify evidence grounding in medical literature.
Judge the machine-generated RoB 2 claim only against the supplied verbatim excerpts from the source trial report. Use these labels exactly:
- Supported: the excerpts directly support the response and its material explanation.
- Contradicted: the excerpts directly conflict with the response or a material explanatory claim.
- Not Found: the claim is relevant to the question, but the excerpts do not establish either support or contradiction.
- Out of Scope: the claim cannot be evaluated for this question, including a genuinely inapplicable conditional item.
Do not use the human gold label, outside knowledge, or unstated assumptions. For Supported or Contradicted, quote the shortest exact passage that justifies the verdict. For Not Found or Out of Scope, return an empty quote. Return JSON only."""


def _question_text(item: dict[str, Any]) -> str:
    supplied = str(item.get("question_text") or "").strip()
    if supplied:
        return supplied
    qid = str(item.get("question_id") or "").casefold()
    question = QUESTION_BY_ID.get(qid)
    return question.text if question else ""


def _explanation(item: dict[str, Any]) -> str:
    values = [item.get("comment"), item.get("reasoning"), item.get("explanation")]
    return " ".join(str(value).strip() for value in values if str(value or "").strip())


def _gold_response(item: dict[str, Any], human_lookup: dict[str, str]) -> str:
    gold = item.get("gold")
    if isinstance(gold, dict):
        response = str(gold.get("response") or "")
        if response:
            return response
    response = str(item.get("human_response") or "")
    if response:
        return response
    return human_lookup.get(str(item.get("question_id") or "").casefold(), "")


def _load_human_lookup(
    human_root: str | Path | None, topic_slug: str, study: str
) -> tuple[dict[str, str], str]:
    if not human_root:
        return {}, ""
    root = Path(human_root)
    expected = root / topic_slug / study / f"{study}.json"
    candidates = [expected] if expected.is_file() else []
    if not candidates:
        candidates = [
            path for path in root.rglob("*.json")
            if path.stem.casefold() == study.casefold() and topic_slug in path.parts
        ]
    if not candidates:
        return {}, ""
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple human-reference files found for {study}/{topic_slug}: "
            + ", ".join(str(path) for path in candidates)
        )

    items = load_json_items(candidates[0])
    exact: dict[str, str] = {}
    canonical: dict[str, set[str]] = {}
    for item in items:
        response = str(item.get("response") or "").strip()
        if not response:
            continue
        question_id = str(item.get("question_id") or "").casefold()
        if question_id:
            exact[question_id] = response
        canonical_id = str(item.get("canonical_question_id") or "").casefold()
        if canonical_id:
            canonical.setdefault(canonical_id, set()).add(response)
    for question_id, responses in canonical.items():
        if question_id not in exact and len(responses) == 1:
            exact[question_id] = next(iter(responses))
    return exact, str(candidates[0])


def _build_query(item: dict[str, Any]) -> str:
    return " ".join(
        value for value in (
            _question_text(item),
            str(item.get("response") or item.get("machine_response") or "").strip(),
            _explanation(item),
        ) if value
    )


def _judge_prompt(study: str, item: dict[str, Any], retrieved: list[dict[str, Any]]) -> str:
    evidence = "\n\n".join(
        f"[Excerpt {hit['rank']}; BM25={hit['score']:.6f}; pages {hit['page_start']}-{hit['page_end']}]\n{hit['snippet']}"
        for hit in retrieved
    )
    return f"""Study: {study}
Question ID: {item.get('question_id', '')}
Question: {_question_text(item)}
Machine response: {item.get('response', item.get('machine_response', ''))}
Machine explanation: {_explanation(item)}

Retrieved source excerpts:
{evidence}

Return:
{{"verdict": "Supported|Contradicted|Not Found|Out of Scope", "quote": "", "rationale": "one concise sentence"}}
"""


def _normalize_verdict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Judge response must be a JSON object")
    verdict = str(value.get("verdict") or "").strip()
    allowed = {"Supported", "Contradicted", "Not Found", "Out of Scope"}
    if verdict not in allowed:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    quote = str(value.get("quote") or "").strip()
    if verdict in {"Not Found", "Out of Scope"}:
        quote = ""
    return {
        "verdict": verdict,
        "quote": quote,
        "rationale": str(value.get("rationale") or "").strip(),
    }


def _quote_verified(quote: str, retrieved: list[dict[str, Any]]) -> bool | None:
    if not quote:
        return None
    needle = normalize_whitespace(quote).casefold().strip('"')
    return any(needle in normalize_whitespace(hit["snippet"]).casefold() for hit in retrieved)


def _metadata_for_response(path: Path) -> dict[str, Any]:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _study_for(path: Path, items: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    for source in (metadata, items[0] if items else {}):
        study = str(source.get("study") or "").strip()
        if study:
            return study
    return path.parent.name if path.parent.name else path.stem


def _output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / input_path.name
    return output_root / input_path.relative_to(input_root)


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return load_json_items(path)
    except Exception:
        return []


def evaluate_file(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    judge: OpenAIJsonClient,
) -> tuple[int, int]:
    items = load_json_items(input_path)
    items = [item for item in items if item.get("question_id")]
    if not items:
        raise ValueError(f"No RoB response items in {input_path}")
    metadata = _metadata_for_response(input_path)
    study = _study_for(input_path, items, metadata)
    topic_slug = str(items[0].get("topic_slug") or metadata.get("topic_slug") or "")
    human_lookup, human_reference_file = _load_human_lookup(
        args.human_root, topic_slug, study
    )
    pdf_path = find_pdf_for_study(args.pdf_root, study)
    retriever = BM25Retriever(
        chunk_pdf(pdf_path, chunk_size=args.chunk_size, overlap=args.overlap)
    )
    output_path = _output_path(input_path, input_root, output_root)
    existing = [] if args.overwrite else _load_existing(output_path)
    completed_ids = {str(item.get("question_id")) for item in existing if item.get("verdict")}
    results = existing[:]

    pending = [item for item in items if str(item.get("question_id")) not in completed_ids]
    if args.limit_items:
        pending = pending[: args.limit_items]
    for index, item in enumerate(pending, start=1):
        qid = str(item.get("question_id") or "")
        print(f"    [{index}/{len(pending)}] {qid}")
        retrieved = retriever.search(_build_query(item), top_k=args.top_k)
        call = judge.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_judge_prompt(study, item, retrieved),
            schema_name="evidence_verdict",
            schema=VERDICT_JSON_SCHEMA,
        )
        verdict = _normalize_verdict(call.data)
        scores = [float(hit["score"]) for hit in retrieved]
        result = {
            "study": study,
            "json_file": input_path.name,
            "pdf_file": pdf_path.name,
            "question_id": qid,
            "question_text": _question_text(item),
            "machine_response": item.get("response", item.get("machine_response", "")),
            "human_response": _gold_response(item, human_lookup),
            "human_reference_file": human_reference_file,
            "topic": item.get("topic", metadata.get("topic", "")),
            "topic_slug": item.get("topic_slug", metadata.get("topic_slug", "")),
            "generator_model": metadata.get("model", args.generator_model or ""),
            "judge_model": args.judge_model,
            "explanation_used": _explanation(item),
            "bm25_topk_scores": scores,
            "bm25_top_score": max(scores) if scores else None,
            "bm25_mean_topk": statistics.fmean(scores) if scores else None,
            "retrieved": retrieved,
            "verdict": verdict,
            "quote_verified_in_retrieved": _quote_verified(verdict["quote"], retrieved),
            "judge_response_id": call.response_id,
            "judge_output_mode": call.output_mode,
        }
        results.append(result)
        atomic_write_json(output_path, results)
    return len(pending), len(items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k PDF evidence with BM25 and assign item-level grounding verdicts."
    )
    parser.add_argument("--responses", "--json", dest="responses", required=True, help="RoB response JSON file or root")
    parser.add_argument("--pdf-root", "--pdf_root", dest="pdf_root", required=True)
    parser.add_argument("--output", "--out-json", "--out_json", dest="output", required=True)
    parser.add_argument(
        "--judge-model",
        "--model",
        dest="judge_model",
        default="gpt-5",
        help="OpenAI judge model ID, including gpt-5 or o3-mini",
    )
    parser.add_argument("--generator-model", help="Used only when response metadata does not identify it")
    parser.add_argument(
        "--human-root",
        help="Optional extracted-human JSON root; labels are retained in output but hidden from the judge",
    )
    parser.add_argument("--api-key-file", "--api_key", dest="api_key_file", help="Otherwise use OPENAI_API_KEY")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--limit-items", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.responses)
    files = find_response_jsons(input_root)
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        raise ValueError(f"No response JSON files found under {input_root}")
    if args.dry_run:
        for path in files:
            items = load_json_items(path)
            metadata = _metadata_for_response(path)
            study = _study_for(path, items, metadata)
            pdf = find_pdf_for_study(args.pdf_root, study)
            print(f"{path}\t{study}\t{pdf}")
        print(f"Validated {len(files)} response file(s).")
        return 0

    judge = OpenAIJsonClient(
        api_key=load_api_key(args.api_key_file),
        model=args.judge_model,
        reasoning_effort=args.reasoning_effort,
    )
    failures = 0
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}")
        try:
            processed, total = evaluate_file(
                path, input_root, Path(args.output), args, judge
            )
            print(f"  processed {processed}; total items {total}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
