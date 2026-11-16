
from affec.data.loader import AFFECDataLoader
from affec.features.extractor import ActionUnitFeatureExtractor
import pandas as pd

# Initialize the authors' data loader
loader = AFFECDataLoader()

# Start with one participant and one run
participant = "sub-aerj"
run = 0

# Load the trials and their emotion labels
trials = loader.merge_trial_data(participant, run)

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