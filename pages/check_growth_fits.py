"""Growth fit review page with well-level editor."""

import streamlit as st

from functions.check_growth_fits import (require_plates,
                                         ui_window_fits_well_editor)

if not st.session_state.get("plates"):
    st.info("Add data first by running **Upload and Analyze**.")
    st.stop()

title_col, popover_col = st.columns([9, 2])
with title_col:
    st.title("Check Growth Fits")
with popover_col:
    st.write("")
    with st.popover("Help", width="stretch"):
        st.markdown("""
**Use this page to review the analysis of individual wells**
- Adjust the phase boundaries and Max OD prediction using the sliders
- Click "Delete" to exclude wells from the analysis
- Click "No growth" to set all growth descriptors to 0
- Click "Re-analyze" to analyze the well with your previously selected settings
- Click and drag on the graph select data points to re-calculate the maximum growth rate

""")
        st.info(
            "You can download any plot by clicking the camera icon in the top right corner of the plot."
        )


plates = require_plates()

ui_window_fits_well_editor(plates)
