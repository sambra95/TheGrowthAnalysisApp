"""Growth fit review page with well-level editor."""

import streamlit as st

from functions.check_growth_fits import (require_plates,
                                         ui_window_fits_well_editor)
from functions.ui_components import page_header_with_help

if not st.session_state.get("plates"):
    st.info("Add data first by running **Upload and Analyze**.")
    st.stop()

page_header_with_help(
    "Check Growth Fits",
    """
**Use this page to review the analysis of individual wells**
- Adjust the phase boundaries and Max OD prediction using the sliders
- Click "Delete" to exclude wells from the analysis
- Click "No growth" to set all growth descriptors to 0
- Click "Re-analyze" to analyze the well with your previously selected settings
- Click and drag on the graph select data points to re-calculate the maximum growth rate

💡 **Tip:** You can download any plot by clicking the camera icon in the top right corner of the plot.
""",
)


plates = require_plates()

ui_window_fits_well_editor(plates)
