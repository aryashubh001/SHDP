# Smart Health Disease Prediction (SHDP)

AI-powered early disease detection system for Diabetes, Heart Disease, and Parkinson's.

## 🏥 Overview

SHDP uses machine learning models trained on medical datasets to predict potential health risks. The system provides:

- **3 Disease Predictions**: Diabetes, Heart Disease, Parkinson's
- **Risk Assessment**: Low, Medium, High risk levels
- **Confidence Scores**: ML model confidence percentages
- **Multi-language Support**: English and Hindi

## 🚀 Quick Start

### Frontend (React)

The frontend runs on Lovable at your project URL. No setup needed!

### Backend (Flask)

The backend must be run locally on your machine:

```bash
# Navigate to backend folder
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The backend will start at `http://localhost:5000`

## 📁 Project Structure

```
smart-health-disease-prediction/
│
├── src/                      # React frontend
│   ├── components/           # UI components
│   ├── contexts/             # React contexts
│   ├── pages/                # Page components
│   └── services/             # API services
│
├── backend/                  # Flask backend
│   ├── app.py               # Main Flask app
│   ├── requirements.txt     # Python dependencies
│   ├── utils/
│   │   ├── load_model.py    # Model loading utility
│   │   └── predictor.py     # Prediction logic
│   └── symptoms_mapping.json
│
└── datasets/                 # ML model files
    ├── diabetes_model.pkl
    ├── heart_model.pkl
    ├── parkinson_model.pkl
    └── README.md
```

## 🤖 Machine Learning Models

Place your trained `.pkl` model files in the `datasets/` folder:

| Model File | Disease | Expected Features |
|------------|---------|-------------------|
| diabetes_model.pkl | Diabetes | 8 features (Pregnancies, Glucose, BP, etc.) |
| heart_model.pkl | Heart Disease | 13 features (age, sex, cp, chol, etc.) |
| parkinson_model.pkl | Parkinson's | 22 voice features |

## 🔧 API Endpoints

### Health Check
```
GET /health
```

### Make Prediction
```
POST /predict
Content-Type: application/json

{
  "disease_type": "diabetes" | "heart" | "parkinson",
  "symptoms": [array of numbers]
}
```

Response:
```json
{
  "predicted_disease": "Diabetes Risk Assessment",
  "risk_level": "Medium",
  "confidence": "82.5%"
}
```

## ⚠️ Medical Disclaimer

This prediction system is for **educational purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.

## 🛠️ Technologies

**Frontend:**
- React + Vite
- Tailwind CSS
- Framer Motion
- Lucide React Icons

**Backend:**
- Python Flask
- scikit-learn
- NumPy
- joblib

## 📝 License

MIT License - See LICENSE file for details.
