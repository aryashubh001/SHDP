"""
Model Loading Utility
=====================

Loads pre-trained ML models from the datasets/ folder.
Models are expected to be saved as .pkl files using joblib or pickle.
"""

import os
import joblib

# Map disease types to their model file paths
MODEL_PATHS = {
    'diabetes': 'datasets/diabetes_model.pkl',
    'parkinson': 'datasets/parkinson_model.pkl',
    'heart': 'datasets/heart_model.pkl'
}


def load_model(disease_type: str):
    """
    Load a pre-trained model for the specified disease type.
    
    Args:
        disease_type: One of 'diabetes', 'parkinson', 'heart'
    
    Returns:
        Loaded sklearn model object
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If the disease_type is invalid
    """
    if disease_type not in MODEL_PATHS:
        raise ValueError(f"Invalid disease type: {disease_type}. "
                        f"Must be one of: {list(MODEL_PATHS.keys())}")
    
    model_path = MODEL_PATHS[disease_type]
    
    # Get absolute path relative to project root
    # Assuming backend/ is in the project root alongside datasets/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_path = os.path.join(base_dir, model_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Model file not found: {full_path}\n"
            f"Please ensure you have placed '{os.path.basename(model_path)}' "
            f"in the 'datasets/' folder."
        )
    
    try:
        model = joblib.load(full_path)
        print(f"[✓] Loaded model: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


def get_model_info(disease_type: str) -> dict:
    """Get information about a model file."""
    if disease_type not in MODEL_PATHS:
        return {'exists': False, 'path': None}
    
    model_path = MODEL_PATHS[disease_type]
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_path = os.path.join(base_dir, model_path)
    
    return {
        'exists': os.path.exists(full_path),
        'path': full_path,
        'relative_path': model_path
    }
