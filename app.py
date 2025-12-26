import streamlit as st

st.set_page_config(
    page_title="Microtitre Growth Analysis",
    layout="wide",
)

nav = st.navigation(
    [
        st.Page("pages/upload_and_analyse.py", title="Upload & Analyse"),
        st.Page("pages/plate_overviews.py", title="Plate overviews"),
        st.Page("pages/edit_growth_stats.py", title="Edit growth stats"),
        st.Page("pages/download_summaries.py", title="Download summaries"),
    ],
    position="top",
)

nav.run()
