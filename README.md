# SMS Spam Classifier — AVIP 2026 Task 2

A reproducible machine-learning project that classifies SMS messages as **spam** or **ham** using TF-IDF text features and Logistic Regression.

## Dataset
- **Name:** UCI SMS Spam Collection
- **Source:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- The training script downloads the dataset automatically on first run.
- Labels: `ham` (legitimate message) and `spam` (unsolicited message).

## Project structure
```text
AI_2_SpamClassifier_byte/
├── data/                  # downloaded dataset (not committed)
├── src/
│   ├── train.py            # download, preprocess, train, evaluate, save
│   └── predict.py          # inference from the command line
├── models/                 # serialized model (generated)
├── results/                # metrics, confusion matrix, example predictions
├── requirements.txt
└── README.md
```

## Setup
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate
Run from the repository root:

```bash
python src/train.py
```

The script performs basic data cleaning, stratified 80/20 train-test splitting, TF-IDF vectorization (unigrams and bigrams), and Logistic Regression training. It reports accuracy, precision, recall, F1-score and a classification report. It saves:
- `models/spam_classifier.joblib`
- `results/metrics.txt`
- `results/confusion_matrix.png`
- `results/example_predictions.csv`

**Do not report performance values until you have run the script.** Metrics depend on the executed experiment.

## Predict a message
After training:

```bash
python src/predict.py "Congratulations, you won a free prize!"
```

The output includes the predicted label, predicted-label confidence, and spam probability.

## Preprocessing and evaluation notes
- Removes missing and duplicate rows.
- Normalizes label casing and trims whitespace.
- Uses a stratified split to preserve the class balance.
- TF-IDF and classifier are kept in one scikit-learn pipeline to avoid fitting the vectorizer on the test set.
- Spam is treated as the positive class for precision, recall and F1-score.

## AVIP deliverables checklist
- [ ] Public GitHub repository with required naming convention
- [ ] Dataset source and preprocessing documented
- [ ] Training script or notebook
- [ ] Accuracy, precision, recall, F1-score and confusion matrix
- [ ] At least 10 example inputs with labels and confidence scores
- [ ] Serialized model and inference instructions
- [ ] Summary (keep it within 300 words)
