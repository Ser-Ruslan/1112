# Heart Disease Dataset - Maximum Heart Rate Prediction

## 🎯 Gradio Web Application for Linear Regression

This is a web application for predicting the **maximum heart rate (thalach)** during exercise testing using a **Linear Regression model** trained on the Heart Disease Cleveland dataset.

## 📋 Project Description

### Objective
Build a Gradio web interface for predicting maximum heart rate based on patient health indicators using a pre-trained linear regression model.

### Dataset
- **Name:** Heart Disease Cleveland
- **Total Records:** 301 patients
- **Training Set:** 240 samples
- **Test Set:** 61 samples
- **Target Variable:** `thalach` (maximum heart rate achieved)
- **Features:** 11 predictors (4 numeric + 7 categorical)

### Model Performance
| Metric | Value |
|--------|-------|
| **R² Score (Test)** | 0.1884 |
| **RMSE (Test)** | 16.71 bpm |
| **MAE (Test)** | 13.87 bpm |
| **Most Important Feature** | age |

### Features Used

#### Numeric Features (4)
- **age** - Patient age in years
- **trestbps** - Resting blood pressure (mmHg)
- **chol** - Serum cholesterol (mg/dl)
- **oldpeak** - ST depression induced by exercise

#### Categorical Features (7)
- **sex** - Gender (male/female)
- **cp** - Chest pain type (asympt/angina/notang/abnang)
- **fbs** - Fasting blood sugar > 120 mg/dl (true/false)
- **restecg** - Resting electrocardiographic state (norm/hyp/abn)
- **exang** - Exercise induced angina (true/false)
- **slope** - ST segment slope (up/flat/down)
- **thal** - Thalassemia type (norm/fix/rev)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Create virtual environment:**
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Application

**Start the Gradio server:**
```bash
python -m app.app
```

The application will start at **`http://127.0.0.1:7860`**

Open this URL in your browser to access the web interface.

## 📁 Project Structure

```
1112/
├── app/
│   ├── __init__.py              # App package
│   ├── model.py                 # Model loading and prediction module
│   └── app.py                   # Gradio web interface
├── models/
│   ├── regression_model.joblib      # Pre-trained Linear Regression model
│   ├── regression_scaler.joblib     # StandardScaler for feature normalization
│   └── regression_config.json       # Model config, coefficients, and metrics
├── tests/
│   └── test_predict.py          # Unit tests for predictions
├── train_regression_model.py    # Script to retrain the model
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── cleve.mod                    # Original heart disease dataset
```

## 🔧 How to Use

### Web Interface Features

1. **Input Form** - Enter all 11 patient health parameters:
   - 4 numeric fields with validation ranges
   - 7 dropdown fields for categorical values

2. **Prediction Output** - Get predicted maximum heart rate in bpm

3. **Visualization** - See a scatter plot showing:
   - Blue dots: Training data points
   - Red star: Your input point with prediction
   - Green line: Linear regression fit

4. **Model Metrics** - Display:
   - R² score on test set
   - Regression equation in LaTeX format
   - Model performance metrics

### Python API

```python
from app.model import load_model

# Load model
model = load_model()

# Make prediction
result = model.predict(
    age=55,
    trestbps=120,
    chol=200,
    oldpeak=1.5,
    sex='male',
    cp='asympt',
    fbs='false',
    restecg='norm',
    exang='false',
    slope='flat',
    thal='norm'
)

print(f"Predicted heart rate: {result['prediction']:.2f} bpm")

# Get model information
info = model.get_model_info()
r2_score = model.get_r2_score()
equation = model.get_equation_latex()

print(f"R² Score: {r2_score:.4f}")
print(f"Equation: {equation}")
```

## 📊 Model Information

### Linear Regression Equation (LaTeX)

$$y = 149.94 - 9.00 \cdot \text{age} + 2.05 \cdot \text{trestbps} + 1.61 \cdot \text{chol} - 1.53 \cdot \text{oldpeak} - 0.42 \cdot \text{sex} - 0.72 \cdot \text{cp} + 0.61 \cdot \text{fbs} - 0.14 \cdot \text{restecg} - 6.47 \cdot \text{exang} + 5.31 \cdot \text{slope} + 0.17 \cdot \text{thal}$$

### Key Findings

| Feature | Coefficient | Interpretation |
|---------|------------|-----------------|
| **age** | -8.99 | 👑 Most influential: younger → higher max heart rate |
| **exang** | -6.47 | Exercise angina → lower max heart rate |
| **slope** | +5.31 | Upsloping ST → higher max heart rate |
| **trestbps** | +2.05 | Higher BP → slightly higher max heart rate |
| **chol** | +1.61 | Higher cholesterol → slightly higher max heart rate |
| **oldpeak** | -1.53 | ST depression → lower max heart rate |

### Data Preprocessing Pipeline

1. **Categorical Encoding** - Label encoding for 7 categorical features
2. **Missing Value Handling** - Replaced '?' with median values
3. **Outlier Removal** - IQR method (1.5 × IQR)
4. **Feature Scaling** - StandardScaler normalization (mean=0, std=1)
5. **Train/Test Split** - 80/20 with seed=42

## 🧪 Testing

### Run Unit Tests

```bash
# Run all tests
python -m unittest tests.test_predict -v

# Run specific test class
python -m unittest tests.test_predict.TestRegressionModel

# With pytest
pytest tests/
```

### What Tests Cover

- ✅ Model loading and initialization
- ✅ Prediction output format and ranges
- ✅ Feature encoding consistency
- ✅ Prediction reproducibility
- ✅ LaTeX equation generation
- ✅ Model metrics validation

## 📝 Example Inputs

### Healthy Young Patient
```
Age: 30
Resting BP: 110 mmHg
Cholesterol: 180 mg/dl
Old Peak: 0.5
Sex: Male
Chest Pain: Asymptomatic
FBS > 120: False
Resting ECG: Normal
Exercise Angina: False
ST Slope: Upsloping
Thal: Normal
```
**Expected:** ~170-180 bpm (high max heart rate)

### Older Patient with Risk Factors
```
Age: 65
Resting BP: 140 mmHg
Cholesterol: 260 mg/dl
Old Peak: 2.5
Sex: Female
Chest Pain: Typical Angina
FBS > 120: True
Resting ECG: Abnormal
Exercise Angina: True
ST Slope: Flat
Thal: Reversible
```
**Expected:** ~120-130 bpm (lower max heart rate)

## 📚 File Formats

### Model Files (.joblib)

```python
import joblib

# Load model
model = joblib.load('models/regression_model.joblib')
scaler = joblib.load('models/regression_scaler.joblib')

# Use for predictions
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
```

### Configuration (.json)

File: `models/regression_config.json`

Contains:
- Feature column names and types
- Model type and parameters
- Coefficients and intercept
- Performance metrics (R², RMSE, MAE)
- Feature scaling parameters (mean, std)

## 🔄 Retraining the Model

To retrain with updated data:

```bash
python train_regression_model.py
```

This will:
1. Load and preprocess the Heart Disease data
2. Train a new LinearRegression model
3. Save model, scaler, and config files to `models/`
4. Display performance metrics on train/test sets

## ⚠️ Important Notes

1. **Medical Disclaimer**
   - This is an educational model only
   - Do not use for actual medical diagnosis or treatment decisions
   - Always consult with healthcare professionals

2. **Data Limitations**
   - Trained on 301 patients from Cleveland Heart Institute
   - May not generalize to other populations
   - Relatively low R² (0.19) indicates moderate predictive power

3. **Model Improvements**
   - Use ensemble methods (Random Forest, Gradient Boosting)
   - Collect more training data
   - Feature engineering (interactions, polynomial features)
   - Hyperparameter optimization
   - Non-linear models (SVR, Neural Networks)

## 🛠️ Dependencies

```
pandas>=1.5.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scikit-learn>=1.3.0    # Machine learning
joblib>=1.3.0          # Model serialization
gradio>=4.0.0          # Web interface
matplotlib>=3.7.0      # Visualization
```

See `requirements.txt` for complete list.

## 📖 References

- Dataset: [Heart Disease Cleveland](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- Scikit-learn: [Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)
- Gradio: [Getting Started](https://www.gradio.app/)
- Visualization: [Matplotlib](https://matplotlib.org/)

## 👥 Author & Version

**Status:** ✅ Complete  
**Last Updated:** November 25, 2025  
**Python Version:** 3.8+  
**Gradio Version:** 4.0+  

---

**Ready for production use!** The model is trained, saved, and integrated with a web interface.
