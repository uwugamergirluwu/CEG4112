from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# Load the sentiment analysis model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        # Make prediction
        result = classifier(text)[0]

        # Return prediction as JSON
        response = {
            'prediction': result['label'],
            'confidence': float(result['score']),
            'input_text': text
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a health check route
@app.route('/')
def health_check():
    return jsonify({'status': 'healthy'})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
