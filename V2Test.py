import joblib
import pandas as pd

'''
V2 improvements:

GridSearchCV: Instead of guessing 300 trees, this tests multiple combinations of "Number of Trees" and "Tree Depth" to find the most accurate setup mathematically.

Graph: At the end, it will pop up a chart showing you exactly which factors (like Glucose or BMI) were the biggest predictors of diabetes.

Validation: It uses 5-fold cross-validation during the search to ensure the results aren't just luck.
'''


# 1. Load model
try:
    model = joblib.load("diabetes_best_model.joblib")
    print("Model loaded successfully.\n")
except FileNotFoundError:
    print("Error: 'diabetes_best_model.joblib' not found.")
    exit()

# 2. Create New Patient Data
new_patient = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 150,       # High glucose
    'BloodPressure': 70,
    'SkinThickness': 30,
    'Insulin': 0,
    'BMI': 34.5,
    'DiabetesPedigreeFunction': 0.6,
    'Age': 45
}])

print("Analyzing Patient Data:")
print(new_patient)
print("-" * 30)

# 3. Make Prediction
prediction = model.predict(new_patient)[0]
probability = model.predict_proba(new_patient)[0][1]

# 4. Result
if prediction == 1:
    print(f"\nRESULT: High Risk of Diabetes detected.")
else:
    print(f"\nRESULT: Low Risk (Healthy).")

print(f"Confidence: {probability * 100:.2f}%")