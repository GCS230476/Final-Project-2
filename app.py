"""
FX Forecasting -- EUR/USD storytelling dashboard
Run:  python -m streamlit run app.py   (use venv_dl)

The app is a guided walkthrough of the whole project, in the order the
work actually happened: question -> data -> leak -> EDA -> features ->
methodology -> models -> results -> live demo.
"""
import streamlit as st

st.set_page_config(
    page_title="EUR/USD Forecasting",
    page_icon=":material/monitoring:",
    layout="wide",
)

pages = {
    "The story": [
        st.Page("app_pages/01_question.py",
                title="1 · The question", icon=":material/help:", default=True),
        st.Page("app_pages/02_data.py",
                title="2 · The data", icon=":material/database:"),
        st.Page("app_pages/03_leak.py",
                title="3 · The leak investigation", icon=":material/search:"),
        st.Page("app_pages/04_eda.py",
                title="4 · Exploring the data", icon=":material/query_stats:"),
        st.Page("app_pages/05_features.py",
                title="5 · Feature engineering", icon=":material/build:"),
        st.Page("app_pages/06_method.py",
                title="6 · Methodology", icon=":material/rule:"),
        st.Page("app_pages/07_models.py",
                title="7 · The models", icon=":material/psychology:"),
        st.Page("app_pages/08_results.py",
                title="8 · Results and discussion", icon=":material/analytics:"),
    ],
    "Demo": [
        st.Page("app_pages/09_live.py",
                title="9 · Live prediction", icon=":material/bolt:"),
    ],
}

page = st.navigation(pages, position="sidebar")

st.title(page.title)
st.caption("EUR/USD forecasting with ML and deep learning · "
           "Dong Cong Gia Khang · Final Project 2026")

page.run()

st.sidebar.caption("Built on the principle: an ugly number that is true "
                   "beats a beautiful number that is fake.")
