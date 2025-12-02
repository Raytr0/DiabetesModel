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
    model = joblib.load("diabetes_xgboost_model.joblib")
    print("Model loaded successfully.\n")
except FileNotFoundError:
    print("Error: 'diabetes_best_model.joblib' not found.")
    exit()

# 2. Create New Patient Data
new_patient = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 99,
    'BloodPressure': 60,
    'SkinThickness': 17,
    'Insulin': 160,
    'BMI': 36.6,
    'DiabetesPedigreeFunction': 0.453,
    'Age': 21
}])

print("Analyzing Patient Data:")
print(new_patient)
print("-" * 30)

# 3. Make Prediction
probs = model.predict_proba(new_patient)[0] # Returns [prob_healthy, prob_diabetes]
prediction = model.predict(new_patient)[0]

# Get the max probability (the confidence of the winner)
confidence_score = max(probs)

if prediction == 1:
    print(f"\nRESULT: High Risk of Diabetes detected.")
else:
    print(f"\nRESULT: Low Risk (Healthy).")

print(f"Confidence: {confidence_score * 100:.2f}%")