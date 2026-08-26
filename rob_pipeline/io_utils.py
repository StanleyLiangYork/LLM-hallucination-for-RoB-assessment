"""Filesystem, PDF, CSV, JSON, and token-budget helpers."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pdfplumber

from .schema import canonical_topic


ROB_CSV_COLUMNS = (
    "Bias arising from the randomization process (judgement)",
    "Bias arising from the randomization process (support)",
    "Bias due to deviations from intended interventions (judgement)",
    "Bias due to deviations from intended interventions (support)",
    "Bias due to missing outcome data (judgement)",
    "Bias due to missing outcome data (support)",
    "Bias in measurement of the outcome (judgement)",
    "Bias in measurement of the outcome (support)",
    "Bias in selection of the reported result (judgement)",
    "Bias in selection of the reported result (support)",
    "Overall bias (judgement)",
    "Overall bias (support)",
)

MODEL_CONTEXT_LIMITS = {
    "gpt-3.5-turbo": 16_385,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o3-mini": 200_000,
    "o3": 200_000,
    "gpt-5": 400_000,
}


@dataclass(frozen=True)
class StudyTopicRecord:
    study: str
    topic_slug: str
    topic: str
    analysis_names: tuple[str, ...]
    csv_row_numbers: tuple[int, ...]
    row: dict[str, str]


def read_csv_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    csv_path = Path(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise ValueError(f"CSV has no header: {csv_path}")
                return list(reader), list(reader.fieldnames)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Could not decode CSV {csv_path}: {last_error}")


def validate_csv_columns(fieldnames: Iterable[str], required: Iterable[str]) -> None:
    fields = set(fieldnames)
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))


def deduplicate_study_topics(rows: list[dict[str, str]]) -> list[StudyTopicRecord]:
    """Collapse repeated subgroup rows to the study/topic pairs used in the paper."""
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, str], str, str]]] = {}
    for row_number, row in enumerate(rows, start=2):
        study = (row.get("Study") or "").strip()
        analysis_name = (row.get("Analysis name") or "").strip()
        if not study or not analysis_name:
            continue
        topic_slug, topic = canonical_topic(analysis_name)
        grouped.setdefault((study, topic_slug), []).append(
            (row_number, row, analysis_name, topic)
        )

    records: list[StudyTopicRecord] = []
    for (study, topic_slug), members in sorted(grouped.items()):
        signatures = {
            tuple((row.get(column) or "").strip() for column in ROB_CSV_COLUMNS)
            for _, row, _, _ in members
        }
        if len(signatures) != 1:
            row_numbers = ", ".join(str(number) for number, *_ in members)
            raise ValueError(
                f"Conflicting RoB annotations for {study}/{topic_slug} at CSV rows {row_numbers}"
            )
        records.append(
            StudyTopicRecord(
                study=study,
                topic_slug=topic_slug,
                topic=members[0][3],
                analysis_names=tuple(dict.fromkeys(name for _, _, name, _ in members)),
                csv_row_numbers=tuple(number for number, *_ in members),
                row=members[0][1],
            )
        )
    return records


def safe_component(value: str) -> str:
    component = re.sub(r"[\\/:*?\"<>|]+", "_", value).strip().strip(".")
    if not component:
        raise ValueError(f"Unsafe empty path component from {value!r}")
    return component


def _name_keys(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_value = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    folded = ascii_value.casefold()
    keys = {re.sub(r"[^a-z0-9]+", "", folded)}
    if re.match(r"^o['’`]", folded):
        keys.add(re.sub(r"[^a-z0-9]+", "", folded[2:]))
    return {key for key in keys if key}


def find_pdf_for_study(pdf_root: str | Path, study: str) -> Path:
    root = Path(pdf_root)
    direct = root / study
    candidates: list[Path] = []
    if direct.is_dir():
        candidates.extend(sorted(direct.glob("*.pdf")))
        candidates.extend(sorted(direct.glob("*.PDF")))

    if not candidates:
        study_keys = _name_keys(study)
        for folder in root.iterdir() if root.is_dir() else ():
            if folder.is_dir() and _name_keys(folder.name).intersection(study_keys):
                candidates.extend(sorted(folder.glob("*.pdf")))
                candidates.extend(sorted(folder.glob("*.PDF")))

    if not candidates:
        expected = root / study / f"{study.replace(' ', '_')}.pdf"
        raise FileNotFoundError(f"No PDF found for {study!r}; expected near {expected}")
    if len({path.resolve() for path in candidates}) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple PDFs found for {study!r}: {names}")
    return candidates[0]


def read_pdf_text(path: str | Path) -> tuple[str, int]:
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages), len(pages)


def load_api_key(api_key_file: str | Path | None) -> str:
    if api_key_file:
        key = Path(api_key_file).read_text(encoding="utf-8").strip()
    else:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "Set OPENAI_API_KEY or provide --api-key-file. Never commit API keys to the repository."
        )
    return key


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, destination)


def load_json_items(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a JSON list of objects in {path}")
    return data


def find_response_jsons(root: str | Path) -> list[Path]:
    path = Path(root)
    if path.is_file():
        return [path]
    files = []
    report_names = {
        "metadata.json",
        "summary.json",
        "overall_summary.json",
        "progress_summary.json",
    }
    for candidate in path.rglob("*.json"):
        if (
            candidate.name.startswith(".")
            or candidate.name in report_names
            or any(part in {"reports", "tables"} for part in candidate.relative_to(path).parts[:-1])
        ):
            continue
        files.append(candidate)
    return sorted(files)


def token_count(text: str, model: str) -> int:
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return max(1, len(text) // 4)


def truncate_to_token_budget(text: str, model: str, budget: int) -> tuple[str, int, bool]:
    if budget <= 0:
        return "", 0, bool(text)
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= budget:
            return text, len(tokens), False
        head_count = int(budget * 0.85)
        tail_count = budget - head_count
        marker = "\n\n[... middle of report omitted to fit model context ...]\n\n"
        shortened = encoding.decode(tokens[:head_count]) + marker + encoding.decode(tokens[-tail_count:])
        return shortened, budget, True
    except ImportError:
        char_budget = budget * 4
        if len(text) <= char_budget:
            return text, max(1, len(text) // 4), False
        head_count = int(char_budget * 0.85)
        tail_count = char_budget - head_count
        marker = "\n\n[... middle of report omitted to fit model context ...]\n\n"
        shortened = text[:head_count] + marker + text[-tail_count:]
        return shortened, budget, True


def context_limit_for_model(model: str, override: int | None = None) -> int:
    if override:
        return override
    base = model.casefold().strip()
    for prefix, limit in MODEL_CONTEXT_LIMITS.items():
        if base == prefix or base.startswith(prefix + "-"):
            return limit
    return 16_385
