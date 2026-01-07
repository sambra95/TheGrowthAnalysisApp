"""Create interactive visualizations of growth data."""

import streamlit as st
from functions.visualization_functions import require_plates, ui_growth_summaries

title_col, popover_col = st.columns([9, 2])
with title_col:
    st.title("Create Visualizations")
with popover_col:
    st.write("")
    with st.popover("Explain this page to me", use_container_width=True):
        st.markdown(
            """
**Actions you can perform on this page:**
- Create custom visualizations of growth parameters (max growth rate, lag time, yield, etc.)
- Group and compare samples by strain or condition
- Generate interactive plots with color-coded legends
- Visualize statistical summaries across experimental conditions

💡 **Tip:** You can download any plot by clicking the camera icon in the top right corner of the plot.
"""
        )

plates = require_plates()

ui_growth_summaries(plates)
