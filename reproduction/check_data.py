
import pandas as pd
from pathlib import Path

# Load the extracted features
df = pd.read_csv("data/processed/au_features.csv")

# Select AU feature columns
au_columns = [col for col in df.columns if col.startswith("AU")]

print("Dataset shape:", df.shape)
print("Number of AU features:", len(au_columns))


# Check missing values

# Find trials where all AU features are missing
missing_au = df[au_columns].isna().all(axis=1)

print("\nTotal trials:", len(df))
print("Trials without AU features:", missing_au.sum())
print("Trials with AU features:", (~missing_au).sum())

# Check if some trials have only a few missing features
partial_missing = (
    df[au_columns].isna().any(axis=1) & ~missing_au
)

print("Trials with partially missing AU features:", partial_missing.sum())


# Check duplicates

print("\nDuplicate rows:", df.duplicated().sum())

# Check duplicates among trials with all AU features
complete_df = df[~df[au_columns].isna().any(axis=1)].copy()

print("Duplicates among complete trials:", complete_df.duplicated().sum())


# Check where AU features are missing

# Count missing trials for each participant and run
missing_by_run = (
    df.loc[missing_au]
    .groupby(["participant", "run"])
    .size()
)

print("\nMissing AU trials by participant and run:")
print(missing_by_run.to_string())


# Check original files

# See if missing features are also missing in the raw data
participants = df.loc[missing_au, "participant"].unique()

print("\nChecking original videostream files:")

for participant in participants:
    folder = Path("data/raw") / participant / "beh"
    files = sorted(folder.glob("*videostream*.tsv.gz"))

    for file in files:
        raw = pd.read_csv(file, sep="\t", header=None)

        # First two columns are timestamps
        measurements = raw.iloc[:, 2:]

        # Count columns that contain data
        available = measurements.notna().any().sum()

        # Get run number from filename
        run = int(file.name.split("_run-")[1].split("_")[0])

        missing_count = missing_by_run.get((participant, run), 0)

        print(
            f"{participant}, run {run}: "
            f"{raw.shape[1]} columns, "
            f"{available} columns with data, "
            f"{missing_count} missing AU trials"
        )


# Summary

print("\nSummary:")
print("Total trials:", len(df))
print("Trials without AU features:", missing_au.sum())
print("Trials with complete AU features:", len(complete_df))
print("Partially missing trials:", partial_missing.sum())
print("Duplicates among complete trials:", complete_df.duplicated().sum())