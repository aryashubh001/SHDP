"""
Smart Health Disease Prediction - Flask Backend
================================================

IMPORTANT: Run this backend server locally on your machine.
Lovable cannot execute Python/Flask code.

Usage:
1. Install dependencies: pip install -r requirements.txt
2. Place your .pkl model files in the datasets/ folder
3. Run: python app.py
4. Backend will be available at http://localhost:5000

"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from utils.load_model import load_model
from utils.predictor import predict_disease

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'SHDP Backend is running'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    
    Request JSON:
    {
        "disease_type": "diabetes" | "parkinson" | "heart",
        "symptoms": [list of numeric values]
    }
    
    Response JSON:
    {
        "predicted_disease": "string",
        "risk_level": "Low" | "Medium" | "High",
        "confidence": "XX%"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        disease_type = data.get('disease_type')
        symptoms = data.get('symptoms')
        
        if not disease_type:
            return jsonify({'error': 'disease_type is required'}), 400
        
        if not symptoms or not isinstance(symptoms, list):
            return jsonify({'error': 'symptoms must be a list of numbers'}), 400
        
        # Validate disease type
        valid_diseases = ['diabetes', 'parkinson', 'heart']
        if disease_type not in valid_diseases:
            return jsonify({
                'error': f'Invalid disease_type. Must be one of: {valid_diseases}'
            }), 400
        
        # Load model and make prediction
        model = load_model(disease_type)
        result = predict_disease(model, disease_type, symptoms)
        
        return jsonify(result)
    
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'available_models': [
            {'type': 'diabetes', 'file': 'datasets/diabetes_model.pkl'},
            {'type': 'heart', 'file': 'datasets/heart_model.pkl'},
            {'type': 'parkinson', 'file': 'datasets/parkinson_model.pkl'}
        ]
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("SHDP Backend Server")
    print("="*50)
    print("Server starting at http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health  - Health check")
    print("  POST /predict - Make prediction")
    print("  GET  /models  - List available models")
    print("\nMake sure your .pkl files are in the datasets/ folder!")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
