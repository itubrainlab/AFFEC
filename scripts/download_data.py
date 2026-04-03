"""Download the AFFEC dataset from Zenodo record 14794876."""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request, urlretrieve


RECORD_ID = "14794876"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"
SPLIT_MODALITY_DIRS = {"core", "eeg", "gaze", "gsr", "pupil", "videostream", "cursor"}
REQUIRED_ROOT_FILES = {
    "dataset_description.json",
    "participants.tsv",
    "participants.json",
    "task-fer_events.json",
    "task-fer_beh.json",
}


def download_record_files(output_dir: Path) -> list[Path]:
    """Download every file listed in the Zenodo record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    request = Request(API_URL, headers={"User-Agent": "AFFEC-downloader/1.0"})
    with urlopen(request) as response:
        record = json.loads(response.read().decode("utf-8"))

    downloaded: list[Path] = []
    for file_info in record.get("files", []):
        filename = file_info["key"]
        download_url = file_info["links"]["self"]
        target = output_dir / filename
        print(f"Downloading {filename}...")
        urlretrieve(download_url, target)
        downloaded.append(target)
        print(f"Saved to {target}")

    return downloaded


def extract_archives(files: list[Path], output_dir: Path) -> None:
    """Extract archives directly into the dataset root to preserve BIDS layout."""
    for file_path in files:
        if file_path.suffix == ".zip":
            print(f"Extracting {file_path.name} -> {output_dir}")
            with zipfile.ZipFile(file_path, "r") as archive:
                archive.extractall(output_dir)
            continue

        if file_path.name.endswith(".tar.gz") or file_path.suffixes[-2:] == [".tar", ".gz"]:
            print(f"Extracting {file_path.name} -> {output_dir}")
            with tarfile.open(file_path, "r:gz") as archive:
                archive.extractall(output_dir)


def normalize_split_extraction_dirs(output_dir: Path) -> None:
    """Merge previously split extraction folders (core/eeg/gaze/...) into BIDS root."""
    for split_name in SPLIT_MODALITY_DIRS:
        split_dir = output_dir / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            continue

        for item in split_dir.iterdir():
            target = output_dir / item.name
            if item.is_dir():
                if not target.exists():
                    shutil.move(str(item), str(target))
                else:
                    for nested in item.rglob("*"):
                        if nested.is_dir():
                            continue
                        rel = nested.relative_to(item)
                        dst = target / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if not dst.exists():
                            shutil.move(str(nested), str(dst))
            else:
                if not target.exists():
                    shutil.move(str(item), str(target))


def validate_bids_layout(output_dir: Path) -> tuple[bool, list[str]]:
    """Validate the uncompressed AFFEC directory follows expected BIDS-style layout."""
    issues: list[str] = []

    missing_root = sorted([name for name in REQUIRED_ROOT_FILES if not (output_dir / name).exists()])
    if missing_root:
        issues.append(f"Missing root files: {', '.join(missing_root)}")

    subjects = sorted([p for p in output_dir.glob("sub-*") if p.is_dir()])
    if not subjects:
        issues.append("No subject folders found (expected sub-*/)")
        return False, issues

    missing_beh = [p.name for p in subjects if not (p / "beh").exists()]
    missing_eeg = [p.name for p in subjects if not (p / "eeg").exists()]
    if missing_beh:
        issues.append(f"Subjects missing beh/: {', '.join(missing_beh[:10])}{' ...' if len(missing_beh) > 10 else ''}")
    if missing_eeg:
        issues.append(f"Subjects missing eeg/: {', '.join(missing_eeg[:10])}{' ...' if len(missing_eeg) > 10 else ''}")

    eventless = [p.name for p in subjects if not list(p.glob(f"{p.name}_task-fer_run-*_events.tsv"))]
    if eventless:
        issues.append(f"Subjects missing run event files: {', '.join(eventless[:10])}{' ...' if len(eventless) > 10 else ''}")

    return len(issues) == 0, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate AFFEC data layout")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading and only normalize/validate local files")
    parser.add_argument("--validate-only", action="store_true", help="Only validate current folder layout")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data" / "raw"

    if not args.validate_only:
        if args.skip_download:
            files = sorted(output_dir.glob("*.zip")) + sorted(output_dir.glob("*.tar.gz"))
            print(f"Using existing archives: {len(files)} file(s)")
        else:
            files = download_record_files(output_dir)
        extract_archives(files, output_dir)

    normalize_split_extraction_dirs(output_dir)
    ok, issues = validate_bids_layout(output_dir)

    if not ok:
        print("BIDS layout validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    n_subjects = len([p for p in output_dir.glob("sub-*") if p.is_dir()])
    print(f"BIDS layout validated: {n_subjects} subject folders found at {output_dir}")


if __name__ == "__main__":
    main()
