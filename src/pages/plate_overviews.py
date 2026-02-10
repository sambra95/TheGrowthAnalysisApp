"""Plate overview page: replicates and plate-level fits."""

import streamlit as st

from src.functions.common import require_plates
from src.functions.ui_components import page_header_with_help
from src.ui_functions.plate_overviews_ui import (
    ui_replicates,
    ui_window_fits_plate_overview,
)

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

ui_replicates(plates)

st.divider()

ui_window_fits_plate_overview(plates)
