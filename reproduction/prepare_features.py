
from affec.data.loader import AFFECDataLoader
from affec.features.extractor import ActionUnitFeatureExtractor
import pandas as pd
from pathlib import Path

# Initialize the authors' data loader
loader = AFFECDataLoader()


from pathlib import Path

# Find all participant folders in the dataset
data_dir = Path("data/raw")
participants = sorted(
    folder.name for folder in data_dir.glob("sub-*")
    if folder.is_dir()
)

print("Number of participants:", len(participants))

# Load trials from all participants and runs
trials = []

for participant in participants:
    for run in range(4):
        run_trials = loader.merge_trial_data(participant, run)
        trials.extend(run_trials)

    print(f"Loaded {participant}")


# Extract facial features for each trial
rows = []

for trial in trials:
    au_data = trial["au_data"]

    if au_data.empty:
        continue

    # Calculate 68 Action Unit features
    features = ActionUnitFeatureExtractor.extract_statistics(au_data)

    # Add participant and emotion labels
    features["participant"] = trial["participant"]
    features["run"] = trial["run"]
    features["perceived_valence"] = trial["perceived_valence"]
    features["felt_arousal"] = trial["felt_arousal"]

    rows.append(features)

# Combine the trials into a table
df = pd.DataFrame(rows)

print("Dataset shape:", df.shape)
print(df.head())


# Check data quality
print("\nDataset quality check:")
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

print("\nMissing values by column:")
print(df.isna().sum()[df.isna().sum() > 0])

# Check trials with missing AU features
missing_au = df["AU01_r_mean"].isna()

print("\nTrials with missing AU features:", missing_au.sum())

print("Duplicates among missing AU trials:",
      df[missing_au].duplicated().sum())

print("Duplicates among complete AU trials:",
      df[~missing_au].duplicated().sum())


# Save extracted features
output_path = Path("data/processed/au_features.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print(f"\nSaved dataset to {output_path}")