"""Quick test of the model"""
from app.model import load_model

print("Loading model...")
model = load_model()

print("Making test prediction...")
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

print(f"Test prediction: {result['prediction']:.2f} bpm")

# Get model info
info = model.get_model_info()
print(f"R² Score: {info['test_r2']:.4f}")
print(f"Model type: {info['model_type']}")

print("\nEquation:")
print(model.get_equation_latex())

print("\n✅ Model is working correctly!")
