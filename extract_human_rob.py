#!/usr/bin/env python3
"""Extract structured human RoB 2 annotations from the Cochrane CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from rob_pipeline.human_parser import parse_human_row
from rob_pipeline.io_utils import (
    ROB_CSV_COLUMNS,
    atomic_write_json,
    deduplicate_study_topics,
    read_csv_rows,
    safe_component,
    validate_csv_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministically parse human RoB 2 support and judgement fields into JSON."
    )
    parser.add_argument("--csv", required=True, help="Cochrane data-row CSV")
    parser.add_argument("--output", required=True, help="Output root")
    parser.add_argument("--topic-slug", help="Optional topic filter, for example Anxiety0_1")
    parser.add_argument("--study-filter", help="Optional case-insensitive study substring")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--column", help=argparse.SUPPRESS)
    parser.add_argument("--api_key", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, fieldnames = read_csv_rows(args.csv)
    validate_csv_columns(fieldnames, ("Study", "Analysis name", *ROB_CSV_COLUMNS))
    records = deduplicate_study_topics(rows)
    output_root = Path(args.output)
    manifest_rows: list[dict[str, Any]] = []

    for record in records:
        if args.topic_slug and record.topic_slug != args.topic_slug:
            continue
        if args.study_filter and args.study_filter.casefold() not in record.study.casefold():
            continue
        folder = output_root / record.topic_slug / safe_component(record.study)
        output_path = folder / f"{safe_component(record.study)}.json"
        if args.dry_run:
            print(f"{record.topic_slug}\t{record.study}\t{output_path}")
            continue
        if output_path.exists() and not args.overwrite:
            status = "skipped"
        else:
            items = parse_human_row(record.row)
            for item in items:
                item.update(
                    {
                        "study": record.study,
                        "topic": record.topic,
                        "topic_slug": record.topic_slug,
                        "analysis_names": list(record.analysis_names),
                        "csv_row_numbers": list(record.csv_row_numbers),
                    }
                )
            atomic_write_json(output_path, items)
            status = "completed"
        manifest_rows.append(
            {
                "status": status,
                "topic_slug": record.topic_slug,
                "topic": record.topic,
                "study": record.study,
                "csv_row_numbers": ";".join(map(str, record.csv_row_numbers)),
                "output": str(output_path),
            }
        )

    if args.dry_run:
        print(f"Validated {len(records)} unique study-topic records.")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "human_extraction_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        fields = ["status", "topic_slug", "topic", "study", "csv_row_numbers", "output"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {sum(row['status'] == 'completed' for row in manifest_rows)} JSON files.")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
