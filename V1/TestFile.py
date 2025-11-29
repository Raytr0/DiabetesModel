import joblib
import pandas as pd

model = joblib.load("diabetes_model.joblib")

new_person = pd.DataFrame([{
    "Pregnancies": 0,
    "Glucose": 118,
    "BloodPressure": 84,
    "SkinThickness": 47,
    "Insulin": 230,
    "BMI": 45.8,
    "DiabetesPedigreeFunction": 0.551,
    "Age": 31
}])

print("Prediction:", model.predict(new_person)[0])  # 1 = diabetic, 0 = not diabetic
