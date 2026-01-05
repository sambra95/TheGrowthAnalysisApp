"""Download processed data tables and plot exports."""
import streamlit as st
from functions.download_summaries import (
    require_plates,
    ui_growth_summaries,
    ui_export,
)

st.title("Download Processed Datasets and Plots")

plates = require_plates()

ui_export(plates)

st.divider()

ui_growth_summaries(plates)
