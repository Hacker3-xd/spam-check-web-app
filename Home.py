import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Spam Classifier", page_icon="📱", layout="wide")

if not st.session_state.get("authenticated", False):
    switch_page("login")

st.title("📱 Spam Classifier Web App")
st.markdown("""
Welcome to the Spam Classifier Web App!

- **Explore** the SMS spam dataset
- **Train** a machine learning model to classify messages
- **Check** if a message or a batch of messages are spam

Navigate using the sidebar to get started.
""")

if st.button("Logout"):
    st.session_state["authenticated"] = False
    switch_page("login")
