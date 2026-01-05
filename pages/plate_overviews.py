"""Plate overview page: replicates and plate-level fits."""
import streamlit as st
from functions.plate_overviews import require_plates, replicates_view, window_plate_view

st.title("Plate Overviews")

plates = require_plates()

replicates_view(plates)

st.divider()

window_plate_view(plates, line_hours=4.0)
