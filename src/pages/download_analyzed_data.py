"""Download processed data tables and plot exports."""

import streamlit as st

from src.functions.common import require_plates
from src.functions.ui_components import page_header_with_help
from src.ui_functions.download_analyzed_data_ui import (
    _render_global_plots_container,
    _render_tabulated_data_container,
    _render_well_level_plots_container,
    get_export_zip,
)

page_header_with_help(
    "Download Analyzed Data",
    """
**Actions you can perform on this page:**
- Download processed growth data including baseline corrected measurements and summary statistics (e.g. max growth rate) tables as CSV files
- Bulk download annotated growth curves and other plots for all or a subset of wells
""",
)

plates = require_plates()
plate_ids = list(plates.keys())

# Initialize session state for ZIP bytes
if "export_zip_bytes" not in st.session_state:
    st.session_state.export_zip_bytes = None

# ---- Plot Options in 2x2 Grid ----
row1_col1, row1_col2 = st.columns(2)

# Tables
with row1_col1:
    (
        c_baseline_corrected,
        c_stats_per_well,
        c_stats_per_sample,
        c_params,
    ) = _render_tabulated_data_container()

# Well Level Plots
with row1_col2:
    (
        c_well,
        c_add_annotations,
        annot_phase,
        annot_umax_point,
        annot_od_max,
        annot_baseline_od,
        annot_tangent,
        annot_fitted_model,
        well_width,
        well_height,
        well_graphs,
        selected_plate_ids,
        wells_by_plate,
    ) = _render_well_level_plots_container(plates, plate_ids)

# Global Plots
with row1_col1:
    (
        c_base,
        c_plate,
        c_replicates,
        global_width,
        global_height,
    ) = _render_global_plots_container()


# Convert lists/dicts to tuples for caching
wells_tuple = tuple((k, tuple(v)) for k, v in sorted(wells_by_plate.items()))

# Use data parameter with a callable to build ZIP only when download is clicked
st.download_button(
    "Download Export ZIP",
    data=lambda: get_export_zip(
        plates,
        c_baseline_corrected,
        c_stats_per_well,
        c_stats_per_sample,
        c_params,
        c_plate,
        c_base,
        c_replicates,
        c_well,
        tuple(well_graphs) if well_graphs else (),
        tuple(selected_plate_ids) if selected_plate_ids else (),
        wells_tuple,
        c_add_annotations,
        annot_phase,
        annot_umax_point,
        annot_od_max,
        annot_baseline_od,
        annot_tangent,
        annot_fitted_model,
        global_width,
        global_height,
        well_width,
        well_height,
    ),
    file_name="export.zip",
    mime="application/zip",
    width="stretch",
    type="primary",
)
