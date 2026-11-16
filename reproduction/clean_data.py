import pandas as pd

# Load extracted features
df = pd.read_csv("data/processed/au_features.csv")

# Select AU features
au_columns = [col for col in df.columns if col.startswith("AU")]

# Remove trials with missing AU features
clean_df = df.dropna(subset=au_columns).copy()

# Save the cleaned dataset
clean_df.to_csv("data/processed/au_features_clean.csv", index=False)

print("Original dataset:", df.shape)
print("Clean dataset:", clean_df.shape)