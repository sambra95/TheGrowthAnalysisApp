"""Create interactive visualizations of growth data."""

import streamlit as st

from functions.ui_components import page_header_with_help
from functions.visualization_functions import require_plates, ui_growth_summaries

if not st.session_state.get("plates"):
    st.info("Add data first by running **Upload and Analyze**.")
    st.stop()

page_header_with_help(
    "Create Visualizations",
    """
**Actions you can perform on this page:**
- Create custom visualizations of growth parameters (max growth rate, lag time, yield, etc.)
- Group and compare samples by strain or condition
- Generate interactive plots with color-coded legends
- Visualize statistical summaries across experimental conditions

💡 **Tip:** You can download any plot by clicking the camera icon in the top right corner of the plot.
""",
)

plates = require_plates()

ui_growth_summaries(plates)
