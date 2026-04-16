import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from CSV
data = pd.read_csv("clean_heart_failure_clinical_records.csv")

# Check data format is correct
print("Data shape:", data.shape)
print(data.head())


# Establish eatures + target
X = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]
#Remove time as well since it's not a feature we want to use for prediction (not a clinical predictor)
X = data.drop(["DEATH_EVENT", "time"], axis=1)


# One-hot encoding
# (safe even if most vars are binary)
X_encoded = pd.get_dummies(X, drop_first=True)


# Train / Validation / Test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train size:", X_train.shape)
print("Validation size:", X_val.shape)
print("Test size:", X_test.shape)