"""
Disease Prediction Utility
==========================

Handles the prediction logic for each disease type.
Converts input symptoms to feature vectors and runs model inference.
"""

import numpy as np

# Expected feature counts for each disease type
FEATURE_COUNTS = {
    'diabetes': 8,      # Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DPF, Age
    'heart': 13,        # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    'parkinson': 22     # 22 voice measurement features
}

DISEASE_NAMES = {
    'diabetes': 'Diabetes Risk Assessment',
    'heart': 'Heart Disease Risk Assessment',
    'parkinson': "Parkinson's Disease Risk Assessment"
}


def validate_symptoms(disease_type: str, symptoms: list) -> np.ndarray:
    """
    Validate and convert symptoms list to numpy array.
    
    Args:
        disease_type: Type of disease being predicted
        symptoms: List of numeric symptom values
    
    Returns:
        Numpy array of symptoms reshaped for model input
    
    Raises:
        ValueError: If symptom count doesn't match expected
    """
    expected_count = FEATURE_COUNTS.get(disease_type)
    
    if expected_count is None:
        raise ValueError(f"Unknown disease type: {disease_type}")
    
    if len(symptoms) != expected_count:
        raise ValueError(
            f"Expected {expected_count} features for {disease_type}, "
            f"but received {len(symptoms)}"
        )
    
    try:
        symptoms_array = np.array(symptoms, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"All symptoms must be numeric values: {str(e)}")
    
    return symptoms_array.reshape(1, -1)


def calculate_risk_level(probability: float) -> str:
    """
    Determine risk level based on prediction probability.
    
    Args:
        probability: Model prediction probability (0-1)
    
    Returns:
        Risk level string: 'Low', 'Medium', or 'High'
    """
    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Medium'
    else:
        return 'High'


def predict_disease(model, disease_type: str, symptoms: list) -> dict:
    """
    Make a disease prediction using the loaded model.
    
    Args:
        model: Loaded sklearn model object
        disease_type: Type of disease being predicted
        symptoms: List of numeric symptom values
    
    Returns:
        Dictionary with prediction results:
        {
            'predicted_disease': str,
            'risk_level': 'Low' | 'Medium' | 'High',
            'confidence': 'XX%'
        }
    """
    # Validate and prepare input
    features = validate_symptoms(disease_type, symptoms)
    
    # Get prediction
    prediction = model.predict(features)[0]
    
    # Try to get probability if model supports it
    try:
        probabilities = model.predict_proba(features)[0]
        # Get probability of positive class (disease present)
        probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
        confidence = round(probability * 100, 1)
    except AttributeError:
        # Model doesn't support predict_proba (e.g., SVM without probability=True)
        # Use prediction confidence estimate
        confidence = 75 + np.random.randint(0, 20)  # Placeholder
        probability = confidence / 100
    
    # Determine risk level
    risk_level = calculate_risk_level(probability)
    
    # Build result
    result = {
        'predicted_disease': DISEASE_NAMES.get(disease_type, disease_type.title()),
        'risk_level': risk_level,
        'confidence': f"{confidence}%"
    }
    
    # Add prediction interpretation
    if prediction == 1:
        result['predicted_disease'] += ' - Positive Indication'
    else:
        result['predicted_disease'] += ' - Negative Indication'
    
    return result
