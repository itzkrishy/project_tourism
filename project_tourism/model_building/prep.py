# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("project_tourism/data/tourism.csv")

# Drop id column, as it does not provide meaningful insights
df.drop(columns=["CustomerID"], inplace=True)

# Handle missing values
print("Handle missing values"
)
# For numerical columns, fill with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Handle specific data quality issues (e.g., "Fe Male" should be "Female")
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female', 'Fe male': 'Female'})

# Encode categorical columns
label_encoder = LabelEncoder()
for col in ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 'ProductPitched', 'Designation']:
    df[col] = label_encoder.fit_transform(df[col])

# Target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the processed dataframes to CSV files in the data directory
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
