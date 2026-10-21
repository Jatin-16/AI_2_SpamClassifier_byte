"""Predict whether a message is spam or ham.
Usage: python src/predict.py "Your message here"
"""
from pathlib import Path
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "spam_classifier.joblib"

def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. First run: python src/train.py")
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python src/predict.py "message text"')
    message = " ".join(sys.argv[1:])
    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba([message])[0]
    prediction = int(model.predict([message])[0])
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = probabilities[prediction]
    print(f"Prediction: {label}")
    print(f"Predicted-label confidence: {confidence:.2%}")
    print(f"Spam probability: {probabilities[1]:.2%}")

if __name__ == "__main__":
    main()
