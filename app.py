from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
model_rf = joblib.load('heart_disease.pkl')

@app.route('/')
def index():
    return "Heart Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    data = request.json
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    user_input = {feature: [float(data[feature])] for feature in features}
    user_DF = pd.DataFrame(user_input)

    # Make a prediction using the trained model
    pred_user = model_rf.predict(user_DF)

    # Return the result as JSON
    result = "No heart disease. You're healthy" if pred_user[0] == 0 else "Heart disease. Take precautions"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
