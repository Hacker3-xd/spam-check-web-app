import streamlit as st
import pandas as pd
from visualize import plot_class_distribution, plot_wordcloud
import os

st.header("📊 Data Exploration / EDA")

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

    if uploaded_file is not None:
        return read_csv_with_fallback(uploaded_file)
    elif os.path.exists("spam.csv"):
        return read_csv_with_fallback("spam.csv")
    else:
        return None


uploaded = st.file_uploader(
    "Upload a CSV file (with columns 'label' and 'message') for EDA, or place spam.csv in root:",
    type=["csv"],
)
df = get_dataframe(uploaded)
if df is not None:
    st.write("### Dataset Preview", df.head())
    st.write(f"Total messages: {len(df)}")
    st.write("### Class Distribution")
    st.pyplot(plot_class_distribution(df))
    st.write("### Word Clouds")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Spam Messages")
        spam_words = " ".join(df[df["label"] == "spam"]["message"])
        st.pyplot(plot_wordcloud(spam_words, title="Spam Word Cloud"))
    with col2:
        st.write("Ham Messages")
        ham_words = " ".join(df[df["label"] == "ham"]["message"])
        st.pyplot(plot_wordcloud(ham_words, title="Ham Word Cloud"))
else:
    st.info(
        "Please upload a CSV file or place 'spam.csv' in the root directory with 'label' and 'message' columns."
    )
