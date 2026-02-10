"""UI helpers for the Check Growth Fits page."""

import numpy as np
import pandas as pd
import streamlit as st
import growthcurves.plot as gc_plot
from growthcurves.inference import bad_fit_stats

from functions.check_growth_fits import (
    _add_lasso_selected_points,
    _cycle,
    _delete_well_from_plate,
    _sg_params_for_plate,
    analyse_well,
    update_growth_stats_from_lasso,
    well_order_A1_to_H12,
)
from functions.plotting_functions import _finite_sorted_xy, is_bad_fit, plot_derivative_metric


def _format_growth_stats_table(gs: dict) -> pd.DataFrame:
    """Format growth stats into a displayable table."""
    if not gs or is_bad_fit(gs):
        # Check if there's a reason for the fit failure
        reason = gs.get("no_growth_reason", "--") if gs else "--"
        return pd.DataFrame(
            {"Metric": ["No growth detected", "Reason"], "Value": ["--", reason]}
        )

    # Define metrics to display with nice labels and formatting
    metrics = [
        ("fit_method", "Fit Method", lambda x: str(x) if x else "sliding_window"),
        ("model_rmse", "RMSE", lambda x: f"{float(x):.5f}" if pd.notna(x) else "--"),
        ("max_od", "Maximum OD", lambda x: f"{float(x):.4f}" if pd.notna(x) else "--"),
        (
            "mu_max",
            "Maximum Growth Rate (1/h)",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "intrinsic_growth_rate",
            "Intrinsic Growth Rate (1/h)",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "time_at_umax",
            "Time at Max Growth (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "od_at_umax",
            "OD at Max Growth",
            lambda x: f"{float(x):.4f}" if pd.notna(x) else "--",
        ),
        (
            "exp_phase_start",
            "Lag Phase End (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "exp_phase_end",
            "Exponential Phase End (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "t_window_start",
            "Analysis Window Start (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "t_window_end",
            "Analysis Window End (h)",
            lambda x: f"{float(x):.2f}" if pd.notna(x) else "--",
        ),
        (
            "phase_boundary_method",
            "Phase Boundary Method",
            lambda x: str(x) if x else "--",
        ),
    ]

    rows = []
    for key, label, formatter in metrics:
        value = gs.get(key)
        # Backward compatibility for previously exported/serialized stats.
        if key == "mu_max" and value is None:
            value = gs.get("specific_growth_rate")
        try:
            formatted_value = formatter(value) if value is not None else "--"
        except (ValueError, TypeError):
            formatted_value = "--"
        rows.append({"Metric": label, "Value": formatted_value})

    return pd.DataFrame(rows)


def _phase_controls(plate: dict, well: str, *, key: str):
    """Render phase/OD sliders and actions; writes into growth_stats."""

    processed = (plate.get("processed_data") or {}).get(well)
    if processed is None or processed.empty:
        st.info(f"No data for {well}")
        return np.nan, np.nan, True

    t = processed["Time"]
    t_min, t_max = float(t.min()), float(t.max())
    step = float(max((t_max - t_min) / 200.0, 0.01))

    growth_stats = (plate.get("growth_stats") or {}).setdefault(well, {})
    ss_key = f"phase__{key}"
    maxod_key = f"maxod__{key}"
    lasso_time_key = f"lasso_time__{key}"

    def _sync_widgets_from_growth_stats():
        """Sync widget state from the current growth_stats dict."""
        lag0 = growth_stats.get("exp_phase_start")
        exp0 = growth_stats.get("exp_phase_end")
        lag0 = float(lag0) if pd.notna(lag0) else t_min
        exp0 = float(exp0) if pd.notna(exp0) else t_min
        st.session_state[ss_key] = (lag0, exp0)

        # Clamp max_od to the actual max in the processed data to avoid slider errors
        actual_max_od = float(max(processed["baseline_corrected"]))
        stored_max_od = float(growth_stats.get("max_od", 0.0))
        st.session_state[maxod_key] = min(stored_max_od, actual_max_od)

        # Track the last lasso update time we've synced
        st.session_state[lasso_time_key] = growth_stats.get("_lasso_update_time")

    # Sync widgets if they don't exist OR if growth_stats was updated by lasso selection
    current_lasso_time = growth_stats.get("_lasso_update_time")
    last_synced_time = st.session_state.get(lasso_time_key)

    if ss_key not in st.session_state or current_lasso_time != last_synced_time:
        _sync_widgets_from_growth_stats()

    st.write("")  # just adds some space

    c1, c2 = st.columns(2)
    with c1:
        lag_end, exp_end = st.slider(
            "Set phase boundaries (hours)",
            t_min,
            t_max,
            step=step,
            key=ss_key,
        )

    with c2:
        max_od = st.slider(
            "Set maximum OD",
            0.0,
            max(processed["baseline_corrected"]),
            step=max(processed["baseline_corrected"]) / 120,
            key=maxod_key,
        )

    # Persist boundaries unless we're deleting
    growth_stats["exp_phase_start"] = float(lag_end)
    growth_stats["exp_phase_end"] = float(exp_end)
    growth_stats["max_od"] = float(max_od)

    c1, c2, c3 = st.columns(3)

    def _on_no_growth():
        """Mark the well as no-growth and reset widgets."""
        growth_stats.update(bad_fit_stats())
        growth_stats["no_growth_reason"] = "manually assigned"
        # Clear lasso-specific keys
        growth_stats.pop("_used_fit_times", None)
        growth_stats.pop("_lasso_update_time", None)
        _sync_widgets_from_growth_stats()

    def _on_reanalyse():
        """Re-run analysis for the well and refresh widgets."""
        plate["growth_stats"][well] = analyse_well(plate, well)
        # refresh local ref + sync widget state
        growth_stats.update(plate["growth_stats"][well])
        # Clear lasso-specific keys so all points show as red (original behavior)
        growth_stats.pop("_used_fit_times", None)
        growth_stats.pop("_lasso_update_time", None)
        _sync_widgets_from_growth_stats()

    def _on_delete():
        """Remove the well from the plate and clear widget state."""
        _delete_well_from_plate(plate, well)
        st.session_state.pop(ss_key, None)
        st.session_state.pop(maxod_key, None)
        st.session_state.pop(lasso_time_key, None)

        # Update params to include this well in remove_wells list
        params = plate.setdefault("params", {})
        remove_wells = params.get("remove_wells", False)
        if remove_wells is False or not remove_wells:
            params["remove_wells"] = [well]
        elif well not in remove_wells:
            params["remove_wells"] = list(remove_wells) + [well]

    with c1:
        st.button(
            "No Growth",
            width="stretch",
            type="primary",
            key=f"nogrowth__{key}",
            on_click=_on_no_growth,
        )

    with c2:
        st.button(
            "Re-analyse",
            type="primary",
            width="stretch",
            on_click=_on_reanalyse,
        )

    with c3:
        st.button(
            "Exclude from analysis",
            width="stretch",
            type="tertiary",
            key=f"deletewell__{key}",
            on_click=_on_delete,
        )

    st.write("")  # just adds some space

    return float(lag_end), float(exp_end), False


@st.fragment
def ui_window_fits_well_editor(plates: dict):
    """Render the well editor UI for interactive window fit adjustments."""
    plate_ids = sorted(plates)

    # Initialize plate selection if not set
    if "winfit_plate" not in st.session_state:
        st.session_state["winfit_plate"] = plate_ids[0]

    # Get wells with data from the current plate
    current_plate_id = st.session_state.get("winfit_plate", plate_ids[0])
    current_plate = plates.get(current_plate_id, {})
    processed_data = current_plate.get("processed_data") or {}

    # Get available wells and sort them in A1-H12 order
    all_standard_wells = well_order_A1_to_H12()
    wells = [w for w in all_standard_wells if w in processed_data]

    # If no wells with data, fall back to standard ordering
    if not wells:
        wells = all_standard_wells

    # Ensure the selected well exists in the current plate's wells
    if (
        "winfit_well" not in st.session_state
        or st.session_state["winfit_well"] not in wells
    ):
        st.session_state["winfit_well"] = wells[0]
    st.session_state["winfit_well"]

    def _move_well(step: int):
        """Move the active well forward/backward."""
        st.session_state["winfit_well"] = _cycle(
            wells, st.session_state.get("winfit_well", wells[0]), step
        )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            plate_col, popover_col, toggle_col1 = st.columns(
                [2, 0.9, 0.9], vertical_alignment="bottom", gap="small"
            )
            with plate_col:
                plate_id = st.selectbox("Plate", plate_ids, key="winfit_plate")
            with popover_col:
                with st.popover("Annotations", width="stretch"):
                    show_phase_boundaries = st.toggle(
                        "Phase boundaries",
                        value=st.session_state.get(
                            "show_phase_boundaries_toggle", True
                        ),
                        key="show_phase_boundaries_toggle",
                    )
                    show_umax_point = st.toggle(
                        "Max growth rate point",
                        value=st.session_state.get("show_umax_point_toggle", True),
                        key="show_umax_point_toggle",
                    )
                    show_max_od = st.toggle(
                        "Max OD",
                        value=st.session_state.get("show_max_od_toggle", True),
                        key="show_max_od_toggle",
                    )
                    show_baseline_od = st.toggle(
                        "Baseline OD",
                        value=st.session_state.get("show_baseline_od_toggle", True),
                        key="show_baseline_od_toggle",
                    )
                    show_tangent = st.toggle(
                        "Tangent line at max growth",
                        value=st.session_state.get("show_tangent_toggle", False),
                        key="show_tangent_toggle",
                    )
                    show_fitted_model = st.toggle(
                        "Fitted model curve",
                        value=st.session_state.get("show_fitted_model_toggle", True),
                        key="show_fitted_model_toggle",
                    )
            with toggle_col1:
                log_scale = st.toggle(
                    "Log scale",
                    value=st.session_state.get("log_scale_toggle", False),
                    key="log_scale_toggle",
                )

            prev, mid, next_ = st.columns([2, 4, 2], vertical_alignment="bottom")
            with prev:
                st.button(
                    "",
                    width="stretch",
                    on_click=_move_well,
                    args=(-1,),
                    key="well_prev",
                    shortcut="Left",
                    type="primary",
                )
            with mid:
                well = st.selectbox(
                    "Well",
                    wells,
                    key="winfit_well",
                    index=wells.index(st.session_state["winfit_well"]),
                )
            with next_:
                st.button(
                    "",
                    width="stretch",
                    on_click=_move_well,
                    args=(+1,),
                    key="well_next",
                    shortcut="Right",
                    type="primary",
                )

    with col2:
        with st.container(border=True):
            plate = plates[plate_id]
            key = f"{plate_id}_{well}"

            lag_end, exp_end, no_growth = _phase_controls(plate, well, key=key)
    if no_growth:
        return

    sg_w, sg_p, _ = _sg_params_for_plate(plates, plate_id)
    processed = plate.get("processed_data") or {}
    growth_stats = plate.get("growth_stats") or {}
    fit_parameters = plate.get("fit_parameters") or {}
    gs = growth_stats.get(well) or {}

    # Display growth status indicator and stats table
    status_col, expander_col = st.columns([2, 5])

    with status_col:
        # Visual indicator for growth detection
        if is_bad_fit(gs):
            reason = gs.get("no_growth_reason", "No growth detected")
            st.container(border=True).error(f"**No Growth:** {reason}")
        else:
            st.container(border=True).success("**Growth Detected**")

    with expander_col:
        # Display growth stats table in an expander
        with st.expander(f"Growth Statistics for Well {well}"):
            stats_df = _format_growth_stats_table(gs)
            # Use a key based on growth stats values to force update when they change
            # Include all key metrics that change during lasso selection
            table_key = (
                f"stats_table_{plate_id}_{well}_"
                f"{gs.get('mu_max', gs.get('specific_growth_rate', 0))}_"
                f"{gs.get('max_od', 0)}_"
                f"{gs.get('exp_phase_start', 0)}_"
                f"{gs.get('exp_phase_end', 0)}_"
                f"{gs.get('model_rmse', 0)}_"
                f"{gs.get('_lasso_update_time', '')}"
            )
            st.dataframe(stats_df, width="stretch", hide_index=True, key=table_key)

        st.caption(
            "💡 **Tip:** Click and drag on the growth curve plot below to select a subset of data points. "
            "The analysis will be automatically rerun using only the selected points to recalculate growth parameters."
        )

    st.divider()

    chart_key = f"lasso_fit_{plate_id}_{well}"

    # Get the processed data for this well
    d = processed.get(well)
    if d is not None and not d.empty:
        # Get time and OD data
        t_raw, y_raw = _finite_sorted_xy(
            d["Time"].to_numpy(), d["baseline_corrected"].to_numpy()
        )

        if t_raw.size > 0:
            # Use hours throughout (no display conversion)
            t_display = t_raw

            # Determine scale
            scale = "log" if log_scale else "linear"

            # Create base plot using growthcurves - this matches the notebook pattern
            fig_main = gc_plot.create_base_plot(t_display, y_raw, scale=scale)

            # Highlight lasso-selected points (default: all points)
            selected_times = gs.get("_used_fit_times")
            if not selected_times:
                selected_times = t_raw.tolist()
            fig_main = _add_lasso_selected_points(
                fig_main,
                t_raw,
                y_raw,
                selected_times,
                scale=scale,
            )

            # Annotate plot with growth stats if available
            if not is_bad_fit(gs) and gs:
                # Get fit result from session state
                fit_result = fit_parameters.get(well)

                # Pass the stored growth stats and fit result directly
                # No need to reconstruct - use the original values from the fit
                fig_main = gc_plot.annotate_plot(
                    fig_main,
                    fit_result=fit_result,
                    stats=gs,
                    show_fitted_curve=show_fitted_model,
                    show_phase_boundaries=show_phase_boundaries,
                    show_crosshairs=show_umax_point,
                    show_od_max_line=show_max_od,
                    show_n0_line=show_baseline_od,
                    show_umax_marker=show_umax_point,
                    show_tangent=show_tangent,
                    scale=scale,
                )

            # Update axis labels
            time_label = "Time (hours)"
            y_label = "ln(OD600)" if log_scale else "OD600 (baseline-corrected)"
            # Set x-axis range to exactly match data range (removes gap at y-axis)
            fig_main.update_xaxes(
                title=time_label,
                showgrid=False,
                type="linear",
                range=[float(t_display.min()), float(t_display.max())],
            )
            fig_main.update_yaxes(title=y_label, showgrid=False)

            # Apply layout for lasso selection functionality
            fig_main.update_layout(
                uirevision="keep",
                dragmode="lasso",
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=20, r=20, t=20, b=20),
                height=600,
            )
        else:
            fig_main = go.Figure()
    else:
        fig_main = go.Figure()

    st.plotly_chart(
        fig_main,
        key=chart_key,
        selection_mode="lasso",
        on_select=lambda: update_growth_stats_from_lasso(
            plates, plate_id, well, chart_key
        ),
        width="stretch",
    )

    # Show derivative plots
    fig_dndt = plot_derivative_metric(
        plate, well, metric="dndt", sg_window=sg_w, sg_poly=sg_p, gs=gs
    )
    fig_mu = plot_derivative_metric(
        plate, well, metric="mu", sg_window=sg_w, sg_poly=sg_p, gs=gs
    )
    st.plotly_chart(fig_dndt, width="stretch")
    st.plotly_chart(fig_mu, width="stretch")
