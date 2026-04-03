"""Prepare corrected AFFEC files for Zenodo re-upload.

Produces two outputs inside ``data/zenodo_upload/``:

1. ``videostream.zip``
   Full replacement of the original Zenodo ``videostream.zip``.
   Contains every ``*_recording-videostream_physio.json`` and
   ``*_recording-videostream_physio.tsv.gz`` from ``data/raw/sub-*/beh/``,
   preserving the ``sub-*/beh/<filename>`` internal path structure used in
   the original archive.  ``.bak`` backup files are excluded.

2. ``videostream_corrections_only.zip``
   Lightweight patch archive containing only the 91 corrected JSON sidecars
   (69 standard repairs + 22 forced surrogate-template repairs) taken
   directly from the two ``data/fixes_videostream_json*/`` staging
   directories.  Upload this alongside a changelog note so users of the
   original release can apply the patch without re-downloading 8.3 GB.

Usage
-----
    python scripts/prepare_zenodo_upload.py [--data-root data/raw]
                                            [--fixes-dir data/fixes_videostream_json]
                                            [--fixes-forced-dir data/fixes_videostream_json_forced]
                                            [--out-dir data/zenodo_upload]
                                            [--skip-full]     # skip rebuilding the 8.3 GB zip
                                            [--dry-run]

The script prints progress, a per-subject summary, and a final manifest of
what was packed into each archive.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def collect_videostream_files(data_root: Path) -> List[Tuple[Path, str]]:
    """Return list of (absolute_path, arcname) for all videostream files.

    arcname follows ``sub-*/beh/<filename>`` to match the original zip layout.
    ``.bak`` files are explicitly excluded.
    """
    entries: List[Tuple[Path, str]] = []
    for jp in sorted(data_root.glob("sub-*/beh/*_recording-videostream_physio.*")):
        if jp.suffix == ".bak":
            continue
        # Strip the data_root prefix to get arcname
        arcname = str(jp.relative_to(data_root))
        entries.append((jp, arcname))
    return entries


def collect_corrected_jsons(
    fixes_dir: Path,
    fixes_forced_dir: Path,
) -> List[Tuple[Path, str]]:
    """Return list of (absolute_path, arcname) for corrected JSON sidecars only."""
    entries: List[Tuple[Path, str]] = []
    seen: set = set()

    for base_dir in (fixes_dir, fixes_forced_dir):
        if not base_dir.exists():
            print(f"  [WARN] Fixes directory not found: {base_dir}", file=sys.stderr)
            continue
        for jp in sorted(base_dir.glob("sub-*/beh/*_recording-videostream_physio.json")):
            arcname = f"{jp.parent.parent.name}/beh/{jp.name}"
            if arcname not in seen:
                entries.append((jp, arcname))
                seen.add(arcname)

    return entries


def build_zip(
    entries: List[Tuple[Path, str]],
    out_path: Path,
    label: str,
    dry_run: bool = False,
) -> None:
    total = len(entries)
    total_size = sum(p.stat().st_size for p, _ in entries)
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Building {label}")
    print(f"  → {out_path}")
    print(f"  Files  : {total:,}")
    print(f"  Raw size (uncompressed JSON, compressed TSV): {human_size(total_size)}")

    if dry_run:
        for p, arc in entries[:5]:
            print(f"    {arc}  [{human_size(p.stat().st_size)}]")
        if total > 5:
            print(f"    ... and {total - 5} more")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for i, (fp, arcname) in enumerate(entries, 1):
            zf.write(fp, arcname)
            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                pct = i / total * 100
                print(f"  [{i:4d}/{total}] {pct:5.1f}%  elapsed {elapsed:.0f}s", end="\r")
    elapsed = time.time() - t0
    out_size = out_path.stat().st_size
    print(f"\n  Done in {elapsed:.1f}s  →  {human_size(out_size)}")


def write_manifest(
    entries: List[Tuple[Path, str]],
    out_path: Path,
    label: str,
) -> None:
    """Write a CSV manifest of all files packed into a zip."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arcname", "source_path", "size_bytes"])
        for fp, arc in entries:
            writer.writerow([arc, str(fp), fp.stat().st_size])
    print(f"  Manifest → {out_path}  ({len(entries)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare corrected AFFEC videostream files for Zenodo re-upload."
    )
    parser.add_argument(
        "--data-root",
        default="data/raw",
        help="AFFEC raw data root containing sub-*/ folders (default: data/raw)",
    )
    parser.add_argument(
        "--fixes-dir",
        default="data/fixes_videostream_json",
        help="Directory with standard-repaired JSON sidecars",
    )
    parser.add_argument(
        "--fixes-forced-dir",
        default="data/fixes_videostream_json_forced",
        help="Directory with forced surrogate-template JSON sidecars",
    )
    parser.add_argument(
        "--out-dir",
        default="data/zenodo_upload",
        help="Output directory for produced zip files (default: data/zenodo_upload)",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Skip rebuilding the large full videostream.zip (only build corrections zip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    data_root = (workspace / args.data_root).resolve()
    fixes_dir = (workspace / args.fixes_dir).resolve()
    fixes_forced_dir = (workspace / args.fixes_forced_dir).resolve()
    out_dir = (workspace / args.out_dir).resolve()

    print("=" * 60)
    print("AFFEC Zenodo Upload Preparation")
    print("=" * 60)
    print(f"  data root   : {data_root}")
    print(f"  fixes dir   : {fixes_dir}")
    print(f"  forced dir  : {fixes_forced_dir}")
    print(f"  output dir  : {out_dir}")
    print(f"  skip-full   : {args.skip_full}")
    print(f"  dry-run     : {args.dry_run}")

    # ------------------------------------------------------------------
    # 1. Corrections-only zip  (small, fast)
    # ------------------------------------------------------------------
    corrections = collect_corrected_jsons(fixes_dir, fixes_forced_dir)
    print(f"\nFound {len(corrections)} corrected JSON sidecars")

    corrections_zip = out_dir / "videostream_corrections_only.zip"
    corrections_manifest = out_dir / "videostream_corrections_only_manifest.csv"

    build_zip(corrections, corrections_zip, "videostream_corrections_only.zip", dry_run=args.dry_run)
    if not args.dry_run:
        write_manifest(corrections, corrections_manifest, "corrections manifest")

    # ------------------------------------------------------------------
    # 2. Full videostream.zip  (large, ~8 GB)
    # ------------------------------------------------------------------
    if not args.skip_full:
        all_vs_files = collect_videostream_files(data_root)
        print(f"\nFound {len(all_vs_files)} videostream files in raw data")

        # Sanity: confirm no .bak files slipped through
        bak_count = sum(1 for p, _ in all_vs_files if p.suffix == ".bak")
        if bak_count:
            print(f"  [ERROR] {bak_count} .bak files detected — aborting full zip build.", file=sys.stderr)
            sys.exit(1)

        full_zip = out_dir / "videostream.zip"
        full_manifest = out_dir / "videostream_full_manifest.csv"

        build_zip(all_vs_files, full_zip, "videostream.zip (full replacement)", dry_run=args.dry_run)
        if not args.dry_run:
            write_manifest(all_vs_files, full_manifest, "full manifest")
    else:
        print("\n[SKIP] Full videostream.zip rebuild skipped (--skip-full)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Subjects breakdown
    std_subs = sorted({arc.split("/")[0] for _, arc in corrections
                       if (fixes_dir / arc.split("/")[0]).exists()})
    forced_subs = sorted({arc.split("/")[0] for _, arc in corrections
                          if (fixes_forced_dir / arc.split("/")[0]).exists()})

    print(f"  Standard-repaired subjects ({len(std_subs)}): {', '.join(std_subs)}")
    print(f"  Forced-repaired subjects   ({len(forced_subs)}): {', '.join(forced_subs)}")
    print(f"  Total corrected JSON files : {len(corrections)}")

    if not args.dry_run:
        print(f"\nFiles written to: {out_dir}")
        for f in sorted(out_dir.glob("*")):
            print(f"  {f.name}  [{human_size(f.stat().st_size)}]")

    print("\nNext steps:")
    print("  1. Log in to Zenodo → open your dataset record → 'New version'")
    print("  2. Delete the old videostream.zip from the new version draft")
    print("  3. Upload data/zenodo_upload/videostream.zip  (full corrected archive)")
    print("  4. Also upload videostream_corrections_only.zip as a supplementary file")
    print("  5. Update the dataset description with a correction note")
    print("  6. Publish the new version")


if __name__ == "__main__":
    main()
