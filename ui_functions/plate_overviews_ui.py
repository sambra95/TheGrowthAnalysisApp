"""UI helpers for the Plate Overviews page."""

import streamlit as st

from functions.common import require_plates
from functions.plotting_functions import (
    plot_baseline,
    plot_replicates_by_sample,
    plot_rmse_heatmap,
    plot_window_plate,
)
from functions.ui_components import page_header_with_help


@st.fragment
def ui_replicates(plates: dict):
    """Render a grid of replicate plots by sample."""
    st.subheader("Sample Replicates Across All Plates")
    st.caption("View replicates grouped by sample. Hover over points for details.")
    st.plotly_chart(plot_replicates_by_sample(plates), width="stretch")


@st.cache_data(show_spinner=False)
def _cached_plate_fits_plot(plate_data: dict, sharey: bool = True):
    """Cache the expensive plate fits overview plot generation."""
    return plot_window_plate(plate_data, sharey=sharey)


@st.fragment
def ui_window_fits_plate_overview(plates: dict):
    """Render baseline and plate-window fits for a selected plate."""
    plate_id = st.selectbox("Plate", sorted(plates), key="winfit_plate_overview")
    st.subheader("Plate Blanks")
    st.caption("Baseline values from blank wells. Hover over points for details.")
    plate = plates[plate_id]
    st.plotly_chart(plot_baseline(plate["baseline"], name_by_well=plate.get("name", {})))

    # Header with toggle for shared y-axis
    header_col, toggle_col = st.columns([1, 4], vertical_alignment="center")
    with header_col:
        st.subheader("Plate Fits Overview")
    with toggle_col:
        sharey = st.toggle("Share Y-axis", value=True, key="sharey_toggle")

    st.caption(
        "Growth curve model fits for all wells in the selected plate. "
        "Hover over points for details."
    )
    with st.spinner("Creating Plate Fits Overview..."):
        fig = _cached_plate_fits_plot(plates[plate_id], sharey=sharey)
        st.plotly_chart(fig, width="stretch")

    # Show RMSE heatmap if growth stats are available
    growth_stats = plate.get("growth_stats", {})
    if growth_stats:
        st.subheader("Model Fit Quality (RMSE)")
        st.plotly_chart(
            plot_rmse_heatmap(plate),
            width="stretch",
        )


def render_plate_overviews_page():
    """Render the full Plate Overviews page UI."""
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
