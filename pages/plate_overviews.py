"""Plate overview page: replicates and plate-level fits."""

import streamlit as st
from functions.plate_overviews import require_plates, replicates_view, window_plate_view

title_col, popover_col = st.columns([9, 2])
with title_col:
    st.title("Plate Overviews")
with popover_col:
    st.write("")
    with st.popover("Explain this page to me", width="stretch"):
        st.markdown("""
**Actions you can perform on this page:**
- View grouped replicate growth curves for all your samples and compare to assess consistency
- View OD measurements for the blank wells as well as the mean value that is used for baseline correction
- View the predicted fits on you plate maps to quickly identify problematic wells.

""")

        st.info(
            "You can download any plot by clicking the camera icon in the top right corner of the plot."
        )


plates = require_plates()

replicates_view(plates)

st.divider()

window_plate_view(plates)
