import streamlit as st
import pandas as pd
import os
from model import predict_message, predict_bulk
from docx import Document

st.header("🧪 Spam Checker")

if not st.session_state.get("authenticated", False):
    st.warning("Please select the 'Login' page from the sidebar to log in.")
    st.stop()

MODEL_PATH = "spam_model.joblib"
VECTORIZER_PATH = "vectorizer.joblib"


# Helper to extract messages from different file types
def extract_messages(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except Exception:
            df = pd.read_csv(uploaded_file, encoding="latin-1")
        if "message" in df.columns:
            return df[["message"]]
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        if "message" in df.columns:
            return df[["message"]]
    elif name.endswith(".docx"):
        doc = Document(uploaded_file)
        messages = [p.text for p in doc.paragraphs if p.text.strip()]
        return pd.DataFrame({"message": messages})
    return None


def model_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)


if not model_exists():
    st.info(
        "Please train a model first on the 'Train Model' page before checking for spam."
    )
else:
    option = st.radio(
        "Choose input method:",
        ["Enter a message", "Upload CSV, Excel, or Word file for batch check"],
    )
    if option == "Enter a message":
        message = st.text_area("Enter your message:", height=100)
        if st.button("Check for Spam"):
            if message:
                pred, prob = predict_message(message)
                if pred == "spam":
                    st.error(
                        f"🚫 This message is likely SPAM! (Confidence: {prob:.2%})"
                    )
                else:
                    st.success(
                        f"✅ This message is likely NOT SPAM! (Confidence: {prob:.2%})"
                    )
    else:
        uploaded = st.file_uploader(
            "Upload a CSV, Excel (.xlsx), or Word (.docx) file with a 'message' column or paragraphs:",
            type=["csv", "xlsx", "docx"],
        )
        df = extract_messages(uploaded)
        if df is not None:
            if st.button("Check All for Spam"):
                result_df = predict_bulk(df)
                st.write(result_df[["message", "prediction", "probability"]])
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV", csv, "spam_predictions.csv", "text/csv"
                )
        else:
            st.info(
                "Please upload a valid file with a 'message' column (CSV/Excel) or paragraphs (Word)."
            )
