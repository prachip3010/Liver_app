from flask import Flask, send_file, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler
with open('liver_cirrhosis_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('liver_cirrhosis_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    try:
        return send_file(filename)
    except:
        return "File not found", 404

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json()
        print("Received data:", data)  # Debug log
        
        # Convert data to numpy array
        features = np.array([
            float(data['age']),
            float(data['gender']),
            float(data['albumin']),
            float(data['bilirubin']),
            float(data['cholesterol']),
            float(data['copper']),
            float(data['alk_phos']),
            float(data['sgot']),
            float(data['platelets']),
            float(data['prothrombin']),
            float(data['ascites']),
            float(data['hepatomegaly']),
            float(data['spiders']),
            float(data['edema'])
        ]).reshape(1, -1)
        
        print("Features array:", features)  # Debug log
        
        # Scale the features
        scaled_features = scaler.transform(features)
        print("Scaled features:", scaled_features)  # Debug log
        
        # Get prediction probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        prediction = int(model.predict(scaled_features)[0])
        print("Probabilities:", probabilities)  # Debug log
        print("Prediction:", prediction)  # Debug log
        
        # Return probabilities
        return jsonify({
            'prediction': prediction,  # 0: no cirrhosis, 1: cirrhosis
            'probability': float(probabilities[1])  # Probability of cirrhosis
        })
    except Exception as e:
        print("Error:", str(e))  # Debug log
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
