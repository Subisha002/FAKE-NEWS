from pathlib import Path

import streamlit as st

from utils import get_dashboard_data, predict_news

BASE_DIR = Path(__file__).resolve().parent
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

st.set_page_config(page_title="Fake News Detection Dashboard", layout="wide")

if "admin_authenticated" not in st.session_state:
    st.session_state["admin_authenticated"] = False

if not st.session_state.get("admin_authenticated", False):
    st.title("🔐 Admin Access")
    st.caption("Secure sign-in required to open the fake news detection prediction and analytics workspace.")

    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter Dashboard", use_container_width=True)

    if submitted:
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state["admin_authenticated"] = True
            st.success("Authentication successful. Opening dashboard...")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")
else:
    col_title, col_logout = st.columns([8, 2])
    with col_title:
        st.title("📰 Fake News Detection System")
        st.caption("Professional Streamlit interface for model prediction and analytics review.")
    with col_logout:
        if st.button("Logout", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.rerun()

    tab1, tab2 = st.tabs(["Prediction", "Dashboard"])

    with tab1:
        st.subheader("Analyze a news article")
        news_text = st.text_area(
            "Enter news text",
            height=220,
            placeholder="Paste article text or claim here...",
        )

        if st.button("Analyze News", width="stretch"):
            if news_text.strip():
                label, confidence = predict_news(news_text)
                st.success(f"Prediction: {label}")
                st.metric("Confidence", f"{confidence:.2f}%")
            else:
                st.warning("Please enter some text before analyzing.")

    with tab2:
        st.subheader("Admin Dashboard")
        summary, df, fig = get_dashboard_data()

        if summary is None:
            st.info("No prediction history yet. Run a prediction from the Prediction tab to populate the dashboard.")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Predictions", summary["Total Predictions"])
            col2.metric("Fake News", summary["Fake News"])
            col3.metric("Real News", summary["Real News"])

            if fig is not None:
                st.pyplot(fig)

            st.subheader("Prediction history")
            st.dataframe(df, width="stretch")
