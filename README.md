# Diabetes Prediction Model

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)

## Overview
This project implements a machine learning solution designed to predict the likelihood of diabetes in patients based on diagnostic metrics. The repository documents the full development lifecycle, from the initial baseline model to the final, optimized version.

The objective was to leverage clinical data to accurately classify patients as either diabetic or non-diabetic, aiding in early diagnosis and medical decision-making.

## Repository Structure

The project is organized into versioned directories representing the development stages:

DiabetesModel/  
├── V1/          # Initial prototype and baseline model  
├── V2/          # Final, optimized implementation  
└── README.md    # Project documentation  

## 📂 Dataset

The model was trained and evaluated using the **Pima Indians Diabetes Dataset**.

* **Source:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Description:** This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. All patients here are females at least 21 years old of Pima Indian heritage.
* **Features:** The dataset includes predictors such as Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.

## 📊 Results & Performance

The final model (V2) demonstrated strong predictive capabilities. Below is a detailed breakdown of the performance metrics and feature analysis.

### Performance Metrics

The model achieved a balanced performance between precision and recall, with an **AUC score of 0.80**, indicating good separability between classes.

| Metric | Score | Description |  
| :--- | :--- | :--- |  
| **Accuracy** | **75%** | Overall correctness of the model. |  
| **Precision** | **72%** | Reliability of positive predictions (True Positives / Predicted Positives). |  
| **Recall** | **73%** | Ability to detect true cases (True Positives / Actual Positives). |  
| **F1-Score** | **72%** | Harmonic mean of precision and recall. |  

### Confusion Matrix

The confusion matrix on the test set revealed the following distribution:

| | Predicted Positive | Predicted Negative |  
| :--- | :---: | :---: |  
| **Actual Positive** | **84** (True Positive) | **13** (False Negative) |  
| **Actual Negative** | **27** (False Positive) | **24** (True Negative) |  

### Feature Importance

Analysis of the model reveals which clinical features were most influential in driving predictions. **Glucose levels** and **BMI** were identified as the primary risk factors.

1.  **Glucose** (Highest Impact)
2.  **BMI**
3.  **Age**
4.  **DiabetesPedigreeFunction**
5.  **Insulin**
6.  **Pregnancies**
7.  **BloodPressure**
8.  **SkinThickness** (Lowest Impact)

## Prerequisites

To run the models in this repository, you will need **Python 3.x** and the following standard data science libraries:

    pip install numpy pandas scikit-learn matplotlib seaborn

## Installation & Usage

1. **Clone the repository:**
    
    git clone https://github.com/Raytr0/DiabetesModel.git
    cd DiabetesModel

2. **Navigate to the final version:**
   To use the completed model, navigate to the `V2` directory:

    cd V2

3. **Run the model:**
   Execute the Python script located in the folder (replace `filename.py` with the actual script name):

    python filename.py

## License
This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
