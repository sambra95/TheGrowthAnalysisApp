"""Download processed data tables and plot exports."""
import streamlit as st
from functions.download_summaries import require_plates, ui_export

st.title("Download Analyzed Data")

plates = require_plates()

ui_export(plates)
