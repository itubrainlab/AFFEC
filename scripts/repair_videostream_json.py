"""Repair AFFEC videostream JSON sidecars to match TSV column counts.

This script fixes the known issue where some
`*_recording-videostream_physio.json` files contain an incomplete `Columns`
list (e.g., 190 columns) while the paired `tsv.gz` has the full schema
(e.g., 715 columns).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple


def count_tsv_fields(tsv_gz_path: Path) -> int:
    try:
        with gzip.open(tsv_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
            line = f.readline().rstrip("\n")
        return line.count("\t") + 1 if line else 0
    except Exception:
        return 0


def load_columns(json_path: Path) -> List[str]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cols = data.get("Columns", [])
        return cols if isinstance(cols, list) else []
    except Exception:
        return []


def build_template_headers(data_root: Path) -> Dict[int, List[str]]:
    templates: Dict[int, List[str]] = {}
    for jp in data_root.glob("sub-*/beh/*_recording-videostream_physio.json"):
        cols = load_columns(jp)
        if cols and len(cols) not in templates:
            templates[len(cols)] = cols
    return templates


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair AFFEC videostream JSON sidecars")
    parser.add_argument("--data-root", default="data/raw", help="AFFEC root containing sub-*/")
    parser.add_argument("--apply", action="store_true", help="Write fixes to disk (default is dry-run)")
    parser.add_argument(
        "--force-template-for-unresolved",
        action="store_true",
        help=(
            "For unresolved mismatches (e.g., TSV with 3 fields), force a surrogate "
            "full template (max available Columns schema) so downstream loaders can parse onset/timestamp."
        ),
    )
    parser.add_argument("--report", default="data/raw/videostream_json_repair_report.csv", help="CSV report path")
    parser.add_argument("--backup-ext", default=".bak", help="Backup extension for modified JSON files")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    templates = build_template_headers(data_root)
    max_template_len = max(templates.keys()) if templates else 0
    max_template = templates.get(max_template_len, [])

    rows: List[Tuple[str, int, int, str]] = []
    fixed = 0
    skipped = 0

    for jp in sorted(data_root.glob("sub-*/beh/*_recording-videostream_physio.json")):
        tp = Path(str(jp).replace(".json", ".tsv.gz"))
        if not tp.exists():
            rows.append((str(jp), -1, -1, "missing_tsv"))
            skipped += 1
            continue

        cols = load_columns(jp)
        n_json = len(cols)
        n_tsv = count_tsv_fields(tp)

        if n_json == n_tsv:
            rows.append((str(jp), n_json, n_tsv, "ok"))
            continue

        # Fix only when we have a reliable template for the TSV width.
        template = templates.get(n_tsv)
        if template and len(template) == n_tsv:
            rows.append((str(jp), n_json, n_tsv, "fixed" if args.apply else "would_fix"))
            if args.apply:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                backup_path = jp.with_suffix(jp.suffix + args.backup_ext)
                if not backup_path.exists():
                    backup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                data["Columns"] = template
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                fixed += 1
            continue

        if args.force_template_for_unresolved and max_template:
            rows.append(
                (
                    str(jp),
                    n_json,
                    n_tsv,
                    "forced_surrogate_template" if args.apply else "would_force_surrogate_template",
                )
            )
            if args.apply:
                with open(jp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                backup_path = jp.with_suffix(jp.suffix + args.backup_ext)
                if not backup_path.exists():
                    backup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                data["Columns"] = max_template
                data["_schema_note"] = (
                    "Columns replaced with surrogate full template because paired TSV fields were truncated. "
                    "Raw values should be considered partial for this run."
                )
                with open(jp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                fixed += 1
            continue

        rows.append((str(jp), n_json, n_tsv, "unresolved_no_template"))
        skipped += 1

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["json_path", "json_columns", "tsv_fields", "status"])
        writer.writerows(rows)

    print(f"Report written: {report_path}")
    if args.apply:
        print(f"Fixed files: {fixed}")
    print(f"Skipped/unresolved files: {skipped}")


if __name__ == "__main__":
    main()
