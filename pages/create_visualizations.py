"""Create interactive visualizations of growth data."""

import streamlit as st

from functions.common import require_plates
from functions.ui_components import page_header_with_help
from ui_functions.create_visualizations_ui import (
    ui_growth_curves_controls_container,
    ui_growth_selection_container,
    ui_growth_stats_controls_container,
)

from functions.plotting_functions import (
    plot_mean_growth,
    plot_replicates_scatter,
    plot_single_growth_stat,
)
from functions.visualization_functions import (
    _build_growth_curves_long_df,
    _build_growth_stats_long_df,
    _max_time_hours,
)

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


selection = ui_growth_selection_container(plates)
sel_ids = selection["sel_ids"]
sel_opt = selection["sel_opt"]
sel_sample_names = selection["sel_sample_names"]
has_split = selection["has_split"]

max_t = _max_time_hours(plates)

with st.form(
    "growth_combined_form",
    border=False,
):
    col1, col2 = st.columns([1, 1])
    with col1:
        stats = ui_growth_stats_controls_container(has_split, sel_opt)
    with col2:
        curves = ui_growth_curves_controls_container(max_t, sel_sample_names)

x_col = stats["x_col"]
legend_col = stats["legend_col"]
x_ordered = stats["x_ordered"]
legend_ordered = stats["legend_ordered"]
apply_stats = stats["apply_stats"]

curves_t0 = curves["curves_t0"]
curves_t1 = curves["curves_t1"]
curves_ordered = curves["curves_ordered"]
apply_mean = curves["apply_mean"]
apply_reps = curves["apply_reps"]

# -----------------------------
# Plots
# -----------------------------
if apply_stats:
    long_df, _ = _build_growth_stats_long_df(plates, sel_ids)

    # Display each metric as a separate downloadable plot.
    metrics = [
        "mu_max",
        "intrinsic_growth_rate",
        "doubling_time",
        "max_od",
        "exp_phase_start",
        "exp_phase_end",
        "time_at_umax",
        "od_at_umax",
    ]

    for metric in metrics:
        metric_df = long_df[long_df["metric"] == metric].copy()
        if not metric_df.empty:
            fig = plot_single_growth_stat(
                metric_df,
                x_col=x_col,
                legend_col=legend_col,
                x_order=x_ordered,
                legend_order=legend_ordered,
            )
            st.plotly_chart(fig, width="stretch")

# Build the shared curves DF only if needed
if apply_mean or apply_reps:
    curves_df = _build_growth_curves_long_df(plates, sel_sample_names)

if apply_mean:
    st.plotly_chart(
        plot_mean_growth(curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1),
        width="stretch",
    )

if apply_reps:
    if curves_df.empty:
        st.info("No replicate data found.")
    else:
        st.plotly_chart(
            plot_replicates_scatter(
                curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1
            ),
            width="stretch",
        )
