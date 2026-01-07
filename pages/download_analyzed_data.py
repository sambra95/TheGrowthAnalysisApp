"""Download processed data tables and plot exports."""

import streamlit as st
from functions.export_functions import require_plates, ui_export

st.title("Download Analyzed Data")

plates = require_plates()

ui_export(plates)
