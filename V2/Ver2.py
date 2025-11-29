import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# 1. Load Dataset
# Note: I kept your specific file path.
csv_path = r"C:\Users\Ryan.H\.cache\kagglehub\datasets\ouzgrer\diabetes\versions\1\diabetes.csv"

try:
    df = pd.read_csv(csv_path)
    print("Dataset loaded.")
except FileNotFoundError:
    print("Error: File not found. Please check the path.")
    exit()

# 2. Preprocessing
target_column = "Outcome"
X = df.drop(columns=[target_column])
y = df[target_column]

# Replace zero values
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_as_missing:
    # Only replace 0 if the column exists
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Create Pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing NaNs with Median
    ("scaler", StandardScaler()),                   # Scale features (Good practice)
    ("clf", RandomForestClassifier(random_state=42)) # The Model
])

# 4. Define Hyperparameter Search Space
# This will test 3x3x2 = 18 different model versions to find the best one
param_grid = {
    'clf__n_estimators': [100, 200, 300],    # Number of trees
    'clf__max_depth': [None, 10, 20],        # Maximum depth of trees
    'clf__min_samples_split': [2, 5]         # Min samples required to split a node
}

# cv=5 means 5-Fold Cross Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get the best model from the search
best_model = grid_search.best_estimator_
print(f"\nBest Parameters Found: {grid_search.best_params_}")

# 5. Evaluate Best Model
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

roc_score = roc_auc_score(y_test, y_proba)
print(f"ROC AUC Score: {roc_score:.4f}")

# 6. Feature Importance Visualization
classifier = best_model.named_steps['clf']
importances = classifier.feature_importances_
feature_names = X.columns

# Create a DataFrame for plotting
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
plt.title('Feature Importance: What drives the prediction?')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 7. Save Model
joblib.dump(best_model, "diabetes_best_model.joblib")
print("\nBest model saved as 'diabetes_best_model.joblib'")