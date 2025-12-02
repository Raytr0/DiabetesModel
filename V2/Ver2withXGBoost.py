import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier  # <--- NEW IMPORT
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# 1. Load Dataset
# I kept your specific path here
csv_path = r"C:\Users\Ryan.H\.cache\kagglehub\datasets\ouzgrer\diabetes\versions\1\diabetes.csv"

try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: File not found. Please check the path.")
    exit()

# 2. Preprocessing/Data Cleaning
target_column = "Outcome"
X = df.drop(columns=[target_column])
y = df[target_column]

# Replace zero values with NaN so the Imputer can handle them
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_as_missing:
    if col in X.columns:
        X[col] = X[col].replace(0, np.nan)

# Split data (Stratify ensures we keep the same ratio of diabetic/healthy in train and test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Create Pipeline with XGBoost
# XGBoost is sensitive to scale, so StandardScaler is very important here.
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ))
])

# 4. Define Hyperparameter Search Space
# XGBoost parameters are different from Random Forest:
# learning_rate: How much the model corrects mistakes (Lower = slower but more precise)
# max_depth: XGBoost usually prefers shallow trees (3-6) to avoid overfitting
# scale_pos_weight: Helps if there are fewer 'Diabetic' cases than 'Healthy' ones
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__max_depth': [3, 4, 5],
    'clf__scale_pos_weight': [1, 2]
}

print("\nStarting Grid Search (Training XGBoost)...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

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
# Extracting importance from the XGBoost step inside the pipeline
classifier = best_model.named_steps['clf']
importances = classifier.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='magma') # Changed palette just for fun
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

# 7. Save Model
joblib.dump(best_model, "diabetes_xgboost_model.joblib")
print("\nBest model saved as 'diabetes_xgboost_model.joblib'")