import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extract form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Define feature names (ensure they match training)
            feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']

            # Create DataFrame
            new_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]], columns=feature_names)

            # Scale and predict
            new_data_scaled = standard_scaler.transform(new_data)
            result = ridge_model.predict(new_data_scaled)

            return render_template('home.html', results=result[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Prediction failed. Please check your input.")
    else:
        return render_template('home.html', results=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
