import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Your dataset path
csv_path = r"C:\Users\Ryan.H\.cache\kagglehub\datasets\ouzgrer\diabetes\versions\1\diabetes.csv"

# Load dataset
df = pd.read_csv(csv_path)
print(df.head())
print(df.columns)

# Target column
target_column = "Outcome"
X = df.drop(columns=[target_column])
y = df[target_column]

# Replace zero values with NaN where medically missing
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_as_missing:
    X[col] = X[col].replace(0, np.nan)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Pipeline: Fill missing → Scale → Train model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluate
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nROC AUC Score:", roc_auc_score(y_test, y_proba))

# Save trained model
joblib.dump(model, "diabetes_model.joblib")
print("\n✅ Model saved as diabetes_model.joblib")
