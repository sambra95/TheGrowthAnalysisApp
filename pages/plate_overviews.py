import streamlit as st
from pages.analysis_shared import require_plates, replicates_view, window_plate_view

st.title("Plate Overviews")

plates = require_plates()

st.subheader("Replicates")
replicates_view(plates)

st.divider()

st.subheader("Window fits — Plate view")
window_plate_view(plates, line_hours=4.0)
