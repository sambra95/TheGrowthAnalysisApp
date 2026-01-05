import streamlit as st

st.set_page_config(page_title="TheGrowthAnalysisApp", layout="wide", page_icon="🦠")

nav = st.navigation(
    [
        st.Page("pages/upload_and_analyse.py", title="Upload & Analyse"),
        st.Page("pages/plate_overviews.py", title="Plate Overviews"),
        st.Page("pages/edit_growth_stats.py", title="Check Growth Fits"),
        st.Page("pages/download_summaries.py", title="Download Data and Plots"),
    ],
    position="top",
)

# styles delete buttons in the app
from styling import red_buttons

red_buttons()

nav.run()
