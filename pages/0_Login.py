import streamlit as st

st.set_page_config(
    page_title="Login - Spam Classifier", page_icon="🔒", layout="centered"
)

# 3D-style login box using markdown and CSS
st.markdown(
    """
    <style>
    .login-box {
        background: linear-gradient(145deg, #232526, #414345);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 3rem 2rem 2rem 2rem;
        margin: 3rem auto 2rem auto;
        max-width: 400px;
    }
    .login-title {
        color: #fff;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 8px #00000055;
    }
    .login-footer {
        color: #bbb;
        text-align: center;
        font-size: 1rem;
        margin-top: 1.5rem;
    }
    </style>
    <div class="login-box">
        <div class="login-title">🔒 Spam Classifier Login</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.get("authenticated", False):
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="Enter your email")
        password = st.text_input(
            "Password", type="password", placeholder="Enter your password"
        )
        login_btn = st.form_submit_button("Login")

    if login_btn:
        if email == "hackerx832@gmail.com" and password == "Admin1234!":
            st.session_state["authenticated"] = True
            st.success("Login successful! Use the sidebar to continue.")
        else:
            st.error("Invalid credentials. Please try again.")
else:
    st.success("You are already logged in! Use the sidebar to navigate the app.")
    st.markdown(
        '<div class="login-footer">If you want to log out, please restart the app or clear cookies/session.</div>',
        unsafe_allow_html=True,
    )

if st.session_state.get("authenticated", False):
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.success("You have been logged out. Please log in again to continue.")
        st.stop()
