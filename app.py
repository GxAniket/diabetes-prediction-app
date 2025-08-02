from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# File paths
MODEL_PATH = os.path.join("model.pkl")
SCALER_PATH = os.path.join("scaler.pkl")
CSV_PATH = os.path.join("diabetes.csv")

# Load model and scaler
def load_assets():
    model, scaler = None, None

    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f" Model loaded from {MODEL_PATH}")
        else:
            print(f" Model file not found at {MODEL_PATH}")

        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            print(f" Scaler loaded from {SCALER_PATH}")
        else:
            print(f" Scaler file not found at {SCALER_PATH}")

    except Exception as e:
        print(f" Error loading model or scaler: {e}")

    return model, scaler

# Load once
model, scaler = load_assets()

# Home Page
@app.route("/")
def home():
    return render_template("index.html", prediction=None)

# Single Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        flash("Model or scaler not loaded.", "error")
        return redirect(url_for("home"))

    feature_keys = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    try:
        # Extract inputs
        inputs = []
        for key in feature_keys:
            value = request.form.get(key)
            if not value:
                raise ValueError(f"Missing input for {key}")
            inputs.append(float(value))

        # Convert and scale
        input_df = pd.DataFrame([inputs], columns=feature_keys)
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)[0]
        result_text = "Diabetic" if prediction == 1 else "Not Diabetic"

        flash("Prediction successful!", "success")
        return render_template("index.html", prediction=result_text)

    except ValueError as ve:
        flash(f"Invalid input: {ve}", "error")
        return render_template("index.html", prediction=None)

    except Exception as e:
        flash(f"Unexpected error: {e}", "error")
        return render_template("index.html", prediction=None)

# Batch Prediction from CSV
@app.route("/batch_predict")
def batch_predict():
    if not model or not scaler:
        flash("Model or scaler not loaded.", "error")
        return redirect(url_for("home"))

    try:
        if not os.path.exists(CSV_PATH):
            flash("CSV file not found!", "error")
            return redirect(url_for("home"))

        df = pd.read_csv(CSV_PATH)

        # Check for Outcome column
        if "Outcome" in df.columns:
            X = df.drop("Outcome", axis=1)
        else:
            X = df

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        df["Predicted"] = ["Diabetic" if p == 1 else "Not Diabetic" for p in predictions]

        return render_template(
            "batch_result.html",
            tables=[df.to_html(classes="table table-bordered table-striped", index=False)],
            title="Batch Prediction Results"
        )

    except Exception as e:
        flash(f"Error during batch prediction: {e}", "error")
        return redirect(url_for("home"))

# Run the server
if __name__ == "__main__":
    app.run(debug=True)
