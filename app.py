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

# hacky solution to make all the tertiary buttons red
st.markdown(
    """
    <style>
    /* Destructive tertiary buttons */
    button[kind="tertiary"] {
        background-color: #d32f2f !important;
        color: white !important;
        border: 1px solid #d32f2f !important;
        border-radius: 0.5rem;
        font-weight: 600;
    }

    button[kind="tertiary"]:hover {
        background-color: #b71c1c !important;
        border-color: #b71c1c !important;
        color: white !important;
    }

    button[kind="tertiary"]:focus {
        box-shadow: 0 0 0 0.2rem rgba(211, 47, 47, 0.4);
        outline: none;
    }

    button[kind="tertiary"]:active {
        background-color: #8e0000 !important;
        border-color: #8e0000 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


nav.run()
