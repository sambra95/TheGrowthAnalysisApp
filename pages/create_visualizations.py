"""Create interactive visualizations of growth data."""
import streamlit as st
from functions.visualization_functions import require_plates, ui_growth_summaries

st.title("Create Visualizations")

plates = require_plates()

ui_growth_summaries(plates)
