import streamlit as st
from pages.analysis_shared import require_plates, window_well_view

st.title("Check Growth Fits")

plates = require_plates()

window_well_view(plates, line_hours=4.0)
