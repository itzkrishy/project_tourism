import pandas as pd

RAW_PATH = "project_tourism/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)
print("Dataset loaded successfully.")

# Validate that the expected columns are present before registering it
expected_columns = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'Designation']
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Failure distribution:")
print(df["ProdTaken"].value_counts())
