"""
Train and evaluate a spam SMS classifier.
Dataset: UCI SMS Spam Collection (label, message).
Run from the repository root: python src/train.py
"""
from pathlib import Path
import io
import zipfile
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
for folder in (DATA_DIR, MODEL_DIR, RESULTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = DATA_DIR / "sms_spam_collection.zip"
DATA_PATH = DATA_DIR / "SMSSpamCollection"


def get_dataset():
    """Download the public UCI dataset if it is not already present."""
    if DATA_PATH.exists():
        return
    print("Downloading UCI SMS Spam Collection...")
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    ZIP_PATH.write_bytes(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        member = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
        DATA_PATH.write_bytes(archive.read(member))


def main():
    get_dataset()
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    df = df.dropna(subset=["label", "message"]).drop_duplicates()
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"].str.len() > 0]
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    print(f"Rows after cleaning: {len(df)}")
    print("Class counts:")
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["target"], test_size=0.20,
        random_state=42, stratify=df["target"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=42))
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    with (RESULTS_DIR / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("Test-set metrics (spam is the positive class)\\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\\n")
        f.write("\\nClassification report:\\n")
        f.write(classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0))
    print("\\nTest-set metrics (spam is the positive class):")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\\n", classification_report(y_test, y_pred, target_names=["ham", "spam"], zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("SMS Spam Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    examples = [
        "Congratulations! You have won a free prize. Call now to claim.",
        "Hey, are we still meeting for lunch today?",
        "URGENT! Your account has been selected for a cash reward.",
        "Can you send me the notes from today's class?",
        "Claim your exclusive offer now by clicking this link.",
        "I'll be there in ten minutes.",
        "You have been chosen to receive a free gift voucher.",
        "Please call me when you reach home.",
        "WINNER! You have won a lottery. Reply now.",
        "Thanks for helping me with the assignment.",
        "Your package will arrive tomorrow afternoon.",
        "Get cheap loans instantly. Apply now!",
    ]
    probs = model.predict_proba(examples)[:, 1]
    labels = model.predict(examples)
    predictions = pd.DataFrame({
        "message": examples,
        "predicted_label": ["spam" if label == 1 else "ham" for label in labels],
        "spam_probability": probs.round(4),
        "confidence_of_predicted_label": [
            round(p if label == 1 else 1 - p, 4) for p, label in zip(probs, labels)
        ],
    })
    predictions.to_csv(RESULTS_DIR / "example_predictions.csv", index=False)

    model_path = MODEL_DIR / "spam_classifier.joblib"
    joblib.dump(model, model_path)
    print(f"\\nSaved model: {model_path}")
    print(f"Saved confusion matrix: {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"Saved sample predictions: {RESULTS_DIR / 'example_predictions.csv'}")


if __name__ == "__main__":
    main()
