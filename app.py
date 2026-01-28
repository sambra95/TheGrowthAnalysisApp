import streamlit as st

st.set_page_config(page_title="TheGrowthAnalysisApp", layout="wide", page_icon="logo.svg")

# Display logo (will stay visible even when sidebar is collapsed)
st.logo("logo.svg")

nav = st.navigation(
    [
        st.Page("pages/upload_and_analyse.py", title="Upload & Analyse"),
        st.Page("pages/plate_overviews.py", title="Plate Overviews"),
        st.Page("pages/check_growth_fits.py", title="Check Growth Fits"),
        st.Page("pages/create_visualizations.py", title="Create Visualizations"),
        st.Page("pages/download_analyzed_data.py", title="Download Analyzed Data"),
    ],
    position="top",
)

# styles delete buttons in the app
from styling import red_buttons, green_gradient, green_navbar

red_buttons()
green_gradient()
green_navbar()

nav.run()
