#!/usr/bin/env python3
"""Summarize retrieve-then-verify verdict JSON files into CSV and JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from rob_pipeline.io_utils import find_response_jsons, load_json_items


VERDICTS = ("Supported", "Contradicted", "Not Found", "Out of Scope")


def normalize_verdict(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("verdict") or value.get("label") or ""
    text = str(value or "").strip().casefold().replace("_", " ")
    mapping = {
        "supported": "Supported",
        "contradicted": "Contradicted",
        "not found": "Not Found",
        "notfound": "Not Found",
        "out of scope": "Out of Scope",
        "outofscope": "Out of Scope",
    }
    return mapping.get(text, "Unknown")


def domain_for_question(question_id: str) -> str:
    qid = (question_id or "").strip().casefold().replace(" ", "_")
    for number in range(1, 6):
        if qid.startswith(f"{number}.") or qid.startswith(f"domain_{number}"):
            return f"Domain {number}"
    if qid in {"overall", "overall_bias", "overall_risk_of_bias"}:
        return "Overall"
    return "Other"


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_score(item: dict[str, Any]) -> float | None:
    direct = _float(item.get("bm25_top_score"))
    if direct is not None:
        return direct
    scores = item.get("bm25_topk_scores")
    if isinstance(scores, list):
        values = [_float(value) for value in scores]
        values = [value for value in values if value is not None]
        return max(values) if values else None
    return None


def _mean_topk(item: dict[str, Any]) -> float | None:
    direct = _float(item.get("bm25_mean_topk"))
    if direct is not None:
        return direct
    scores = item.get("bm25_topk_scores")
    if isinstance(scores, list):
        values = [_float(value) for value in scores]
        values = [value for value in values if value is not None]
        return statistics.fmean(values) if values else None
    return None


def _label(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def load_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in find_response_jsons(input_root):
        try:
            items = load_json_items(path)
        except Exception as exc:
            print(f"WARNING: skipped {path}: {exc}")
            continue
        for item in items:
            if not item.get("question_id") or "verdict" not in item:
                continue
            verdict = normalize_verdict(item.get("verdict"))
            machine = str(item.get("machine_response") or item.get("model_response") or "")
            human = str(item.get("human_response") or "")
            rows.append(
                {
                    "source_file": str(path),
                    "study": str(item.get("study") or path.stem),
                    "topic": str(item.get("topic") or ""),
                    "topic_slug": str(item.get("topic_slug") or ""),
                    "generator_model": str(item.get("generator_model") or ""),
                    "judge_model": str(item.get("judge_model") or ""),
                    "question_id": str(item.get("question_id") or ""),
                    "domain": domain_for_question(str(item.get("question_id") or "")),
                    "machine_response": machine,
                    "human_response": human,
                    "verdict": verdict,
                    "is_supported": int(verdict == "Supported"),
                    "is_conservative_hallucination": int(verdict in {"Contradicted", "Not Found"}),
                    "is_strict_hallucination": int(verdict in {"Contradicted", "Not Found", "Out of Scope"}),
                    "agrees_with_human": (
                        int(_label(machine) == _label(human)) if human.strip() else ""
                    ),
                    "bm25_top_score": _top_score(item),
                    "bm25_mean_topk": _mean_topk(item),
                    "quote_verified_in_retrieved": item.get("quote_verified_in_retrieved"),
                }
            )
    return rows


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _sd(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None)


def summarize_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    verdict_counts = Counter(row["verdict"] for row in rows)
    supported = verdict_counts["Supported"]
    conservative = sum(row["is_conservative_hallucination"] for row in rows)
    strict = sum(row["is_strict_hallucination"] for row in rows)
    return {
        "n_items": count,
        "supported_n": supported,
        "contradicted_n": verdict_counts["Contradicted"],
        "not_found_n": verdict_counts["Not Found"],
        "out_of_scope_n": verdict_counts["Out of Scope"],
        "unknown_n": verdict_counts["Unknown"],
        "support_rate": supported / count if count else None,
        "conservative_hallucination_rate": conservative / count if count else None,
        "strict_hallucination_rate": strict / count if count else None,
        "support_percent": 100 * supported / count if count else None,
        "conservative_hallucination_percent": 100 * conservative / count if count else None,
        "strict_hallucination_percent": 100 * strict / count if count else None,
        "mean_bm25_top_score": _mean(row["bm25_top_score"] for row in rows),
        "sd_bm25_top_score": _sd(row["bm25_top_score"] for row in rows),
        "mean_bm25_mean_topk": _mean(row["bm25_mean_topk"] for row in rows),
        "sd_bm25_mean_topk": _sd(row["bm25_mean_topk"] for row in rows),
    }


def grouped_summary(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key) or "") for key in keys)].append(row)
    output = []
    for group_key, members in sorted(groups.items()):
        result = dict(zip(keys, group_key))
        result.update(summarize_block(members))
        output.append(result)
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize hallucination verdict JSON files.")
    parser.add_argument("--input", "--input-dir", "--input_dir", dest="input", required=True)
    parser.add_argument("--output", "--save-csv", "--save_csv", dest="output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_rows(Path(args.input))
    if not rows:
        raise ValueError(f"No verdict items found under {args.input}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    overall = summarize_block(rows)
    by_study = grouped_summary(rows, ("study",))
    by_topic = grouped_summary(rows, ("topic_slug", "topic"))
    by_model = grouped_summary(rows, ("generator_model",))
    by_topic_model = grouped_summary(rows, ("topic_slug", "topic", "generator_model"))
    by_domain = grouped_summary(rows, ("domain", "generator_model"))
    by_verdict = grouped_summary(rows, ("verdict",))

    _write_csv(output / "items.csv", rows)
    _write_csv(output / "by_study.csv", by_study)
    _write_csv(output / "by_topic.csv", by_topic)
    _write_csv(output / "by_model.csv", by_model)
    _write_csv(output / "by_topic_model.csv", by_topic_model)
    _write_csv(output / "by_domain.csv", by_domain)
    _write_csv(output / "bm25_by_verdict.csv", by_verdict)
    _write_csv(output / "overall.csv", [overall])
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "overall": overall,
                "by_topic": by_topic,
                "by_model": by_model,
                "by_domain": by_domain,
                "bm25_by_verdict": by_verdict,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print(f"N items: {overall['n_items']}")
    print(f"Support rate: {overall['support_percent']:.2f}%")
    print(
        "Conservative hallucination rate: "
        f"{overall['conservative_hallucination_percent']:.2f}%"
    )
    print(f"Strict hallucination rate: {overall['strict_hallucination_percent']:.2f}%")
    print(f"Reports written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
