"""Growth fit review page with well-level editor."""
import streamlit as st
from functions.check_growth_fits import require_plates, ui_window_fits_well_editor


st.title("Check Growth Fits")

plates = require_plates()

ui_window_fits_well_editor(plates, line_hours=4.0)
