import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import train_and_save_model
import os

st.header("⚙️ Model Training")

if not st.session_state.get("authenticated", False):
    st.warning("Please select the 'Login' page from the sidebar to log in.")
    st.stop()


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

    def read_excel_with_fallback(path_or_buffer):
        return pd.read_excel(path_or_buffer)

    if uploaded_file is not None:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return read_csv_with_fallback(uploaded_file)
        elif name.endswith(".xlsx"):
            return read_excel_with_fallback(uploaded_file)
    elif os.path.exists("spam.csv"):
        return read_csv_with_fallback("spam.csv")
    else:
        return None


def clean_and_validate_df(df):
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    if "label" not in df.columns or "message" not in df.columns:
        return None
    return df


uploaded = st.file_uploader(
    "Upload a CSV or Excel file (with columns 'label' and 'message') to train, or place spam.csv in root:",
    type=["csv", "xlsx"],
)
df = get_dataframe(uploaded)
df = clean_and_validate_df(df)
if df is not None:
    st.write("### Dataset Preview", df.head())
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            metrics = train_and_save_model(df)
            st.success("Model trained and saved!")
            st.write("### Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['precision']:.2%}")
            col3.metric("Recall", f"{metrics['recall']:.2%}")
            col4.metric("F1 Score", f"{metrics['f1']:.2%}")
            st.write("### Confusion Matrix")
            plt.figure(figsize=(6, 4))
            sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(plt)
else:
    st.info(
        "Please upload a CSV/Excel file or place 'spam.csv' in the root directory with 'label' and 'message' columns."
    )
