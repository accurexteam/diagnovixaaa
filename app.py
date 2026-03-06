from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
import joblib

model = joblib.load("diabetes_model.pkl")
print("MODEL TYPE:", type(model))

import os
print("FLASK FOLDER:", os.getcwd())
print("MODEL TYPE:", type(model))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/diabetes")
def diabetes_page():
    return render_template("diabetes.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    glucose = float(request.form["glucose"])
    hypertension = request.form["hypertension"]
    heart_disease = request.form["heart_disease"]
    hba1c = float(request.form["hba1c"])
    gender = request.form["gender"]
    smoking = request.form["smoking"]


    # Simple encoding example
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0


    gender_male = 1 if gender == "Male" else 0
    gender_other = 1 if gender == "Other" else 0
    smoking_current = 1 if smoking == "Current" else 0
    smoking_ever = 1 if smoking == "Ever" else 0
    smoking_former = 1 if smoking == "Former" else 0
    smoking_never = 1 if smoking == "Never" else 0
    smoking_not_current = 1 if smoking == "Not Current" else 0

    input_data = np.array([[age, bmi,  glucose,hypertension, heart_disease, hba1c,gender_male, gender_other, smoking_current, smoking_ever, smoking_former, smoking_never, smoking_not_current]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        result = "High Risk of Diabetes"
        advice = "Consult a healthcare professional immediately."
    else:
        result = "Low Risk of Diabetes"
        advice = "Maintain healthy lifestyle and regular checkups."

    return render_template("diabetes.html",
                           prediction_text=result,
                           probability=round(probability * 100, 2),
                           advice=advice)

if __name__ == "__main__":
    app.run()