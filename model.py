import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from preprocess import preprocess_text

MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


# Train and save model
def train_and_save_model(df):
    df["processed_message"] = df["message"].apply(preprocess_text)
    X = df["processed_message"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    y_pred = model.predict(X_test_vec)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        "recall": recall_score(y_test, y_pred, pos_label="spam"),
        "f1": f1_score(y_test, y_pred, pos_label="spam"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


# Load model and vectorizer


def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# Predict single message


def predict_message(message):
    model, vectorizer = load_model_and_vectorizer()
    processed = preprocess_text(message)
    vec = vectorizer.transform([processed])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return pred, prob


# Predict for a DataFrame


def predict_bulk(df):
    model, vectorizer = load_model_and_vectorizer()
    df["processed_message"] = df["message"].apply(preprocess_text)
    vec = vectorizer.transform(df["processed_message"])
    preds = model.predict(vec)
    probs = model.predict_proba(vec).max(axis=1)
    df["prediction"] = preds
    df["probability"] = probs
    return df
