"""Plate overview page: replicates and plate-level fits."""

import streamlit as st

from functions.plate_overviews import replicates_view, require_plates, window_plate_view
from functions.ui_components import page_header_with_help

if not st.session_state.get("plates"):
    st.info("Add data first by running **Upload and Analyze**.")
    st.stop()

page_header_with_help(
    "Plate Overviews",
    """
**Actions you can perform on this page:**
- View grouped replicate growth curves for all your samples and compare to assess consistency
- View OD measurements for the blank wells as well as the mean value that is used for baseline correction
- View the predicted fits on you plate maps to quickly identify problematic wells.

💡 **Tip:** You can download any plot by clicking the camera icon in the top right corner of the plot.
""",
)


plates = require_plates()

replicates_view(plates)

st.divider()

window_plate_view(plates)
