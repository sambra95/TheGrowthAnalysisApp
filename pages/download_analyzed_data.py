"""Download processed data tables and plot exports."""

import streamlit as st
from functions.export_functions import require_plates, ui_export

if not st.session_state.get("plates"):
    st.info("Add data first by running **Upload and Analyze**.")
    st.stop()

title_col, popover_col = st.columns([9, 2])
with title_col:
    st.title("Download Analyzed Data")
with popover_col:
    st.write("")
    with st.popover("Help", width="stretch"):
        st.markdown(
            """
**Actions you can perform on this page:**
- Download processed growth data including baseline corrected measurements and summary statistics (e.g. max growth rate) tables as CSV files
- Bulk download annotated growth curves and other plots for all or a subset of wells

"""
        )

plates = require_plates()

ui_export(plates)
