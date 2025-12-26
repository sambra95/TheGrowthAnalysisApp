import streamlit as st
from pages.analysis_shared import require_plates, window_well_view

st.set_page_config(page_title="Edit growth stats", layout="wide")
st.title("Edit growth stats")

plates = require_plates()

window_well_view(plates, line_hours=4.0)
