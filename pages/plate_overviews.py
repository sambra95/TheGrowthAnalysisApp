import streamlit as st
from pages.analysis_shared import require_plates, replicates_view, window_plate_view

st.set_page_config(page_title="Plate overviews", layout="wide")
st.title("Plate overviews")

plates = require_plates()

st.subheader("Replicates")
replicates_view(plates)

st.divider()

st.subheader("Window fits — Plate view")
window_plate_view(plates, line_hours=4.0)
