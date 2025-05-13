import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
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
import os

# Download required NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# Set page config
st.set_page_config(page_title="Spam Classifier", page_icon="📱", layout="wide")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None


def get_dataframe(uploaded_file=None):
    def read_csv_with_fallback(path_or_buffer):
        try:
            df = pd.read_csv(path_or_buffer, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path_or_buffer, encoding="latin-1")
        # Rename columns if needed
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "message"})
        return df

    if uploaded_file is not None:
        return read_csv_with_fallback(uploaded_file)
    elif os.path.exists("spam.csv"):
        return read_csv_with_fallback("spam.csv")
    else:
        return None


def clean_and_validate_df(df):
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    # Check for required columns
    if "label" not in df.columns or "message" not in df.columns:
        return None
    return df


def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Apply stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def train_model(df):
    """Train the spam classifier model"""
    # Preprocess messages
    df["processed_message"] = df["message"].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_message"], df["label"], test_size=0.2, random_state=42
    )

    # Create and fit vectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Calculate metrics
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="spam")
    recall = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")

    return model, vectorizer, accuracy, precision, recall, f1, y_test, y_pred


def main():
    if not st.session_state.get("authenticated", False):
        st.warning("Please select the 'Login' page from the sidebar to log in.")
        st.stop()

    st.title("📱 Spam Classifier")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Choose a page", ["Home", "Data Exploration", "Model Training", "Spam Checker"]
    )

    if page == "Home":
        st.header("Welcome to Spam Classifier!")
        st.write("""
        This application helps you classify text messages as spam or not spam using machine learning.
        
        Features:
        - Data exploration and visualization
        - Machine learning model training
        - Real-time spam classification
        - Model performance metrics
        """)

    elif page == "Data Exploration":
        st.header("Data Exploration")
        df = get_dataframe()
        df = clean_and_validate_df(df)
        if df is not None:
            st.subheader("Dataset Overview")
            st.write(f"Total messages: {len(df)}")
            st.write("Class distribution:")
            st.write(df["label"].value_counts())

            # Create word clouds
            st.subheader("Word Clouds")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Spam Messages")
                spam_words = " ".join(df[df["label"] == "spam"]["message"])
                wordcloud = WordCloud(
                    width=800, height=400, background_color="white"
                ).generate(spam_words)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)

            with col2:
                st.write("Ham Messages")
                ham_words = " ".join(df[df["label"] == "ham"]["message"])
                wordcloud = WordCloud(
                    width=800, height=400, background_color="white"
                ).generate(ham_words)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)

    elif page == "Model Training":
        st.header("Model Training")
        df = get_dataframe()
        df = clean_and_validate_df(df)
        if df is not None:
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    (
                        model,
                        vectorizer,
                        accuracy,
                        precision,
                        recall,
                        f1,
                        y_test,
                        y_pred,
                    ) = train_model(df)
                    st.session_state.model = model
                    st.session_state.vectorizer = vectorizer

                    # Display metrics
                    st.subheader("Model Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.2%}")
                    col2.metric("Precision", f"{precision:.2%}")
                    col3.metric("Recall", f"{recall:.2%}")
                    col4.metric("F1 Score", f"{f1:.2%}")

                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    st.pyplot(plt)

    elif page == "Spam Checker":
        st.header("Spam Checker")
        if st.session_state.model is None:
            st.warning("Please train the model first in the Model Training page!")
        else:
            message = st.text_area("Enter your message:", height=150)
            if st.button("Check for Spam"):
                if message:
                    # Preprocess the message
                    processed_message = preprocess_text(message)
                    # Vectorize
                    message_vec = st.session_state.vectorizer.transform(
                        [processed_message]
                    )
                    # Predict
                    prediction = st.session_state.model.predict(message_vec)[0]
                    probability = st.session_state.model.predict_proba(
                        message_vec
                    ).max()

                    # Display result
                    if prediction == "spam":
                        st.error("🚫 This message is likely SPAM!")
                    else:
                        st.success("✅ This message is likely NOT SPAM!")

                    st.write(f"Confidence: {probability:.2%}")


if __name__ == "__main__":
    main()
