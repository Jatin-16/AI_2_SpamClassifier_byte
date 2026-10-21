# AVIP 2026 — Task 2: Spam SMS Classifier

This project implements a machine-learning classifier to identify SMS messages as spam or legitimate (ham). It uses the UCI SMS Spam Collection dataset and a text-processing pipeline based on Term Frequency–Inverse Document Frequency (TF-IDF) and Logistic Regression.

The dataset is cleaned by removing missing and duplicate records, normalizing labels, and removing empty messages. The data is divided into training and test sets using a stratified 80:20 split. The TF-IDF vectorizer extracts unigram and bigram features, which are used by the classifier to predict message categories.

The model was evaluated on 1,034 test messages. It achieved an accuracy of **97.78%**, spam precision of **90.91%**, spam recall of **91.60%**, and spam F1-score of **91.25%**. The results indicate strong overall classification performance, although some spam messages may still be missed or incorrectly flagged.

The project also includes a prediction script that reports the predicted category, spam probability, and confidence in the predicted category. The trained pipeline can be serialized for reuse without retraining.

The final submission includes the training code, saved model, evaluation metrics, confusion matrix, sample predictions, dependency requirements, and usage documentation.