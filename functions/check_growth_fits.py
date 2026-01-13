"""Interactive well-by-well growth fit inspection and editing UI."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions.data_processing import (
    BAD_FIT,
    _plate_name_map,
    _read_table,
    calculate_growth_descriptors,
    calculate_growth_descriptors_model_based,
    extract_growth_descriptors_from_model,
    fit_growth_model,
)
from functions.plotting_functions import (
    _vlines,
    is_bad_fit,
    plot_model_fit_single,
    plot_window_single,
    plot_window_single_d1,
    plot_window_single_d2,
)


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    """Return plates from session state, or stop with a warning."""
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Selection + stats helpers ----------------
def well_order_A1_to_H12() -> list[str]:
    """Return standard well ordering A1..H12."""
    rows = "ABCDEFGH"
    cols = range(1, 13)
    return [f"{r}{c}" for r in rows for c in cols]  # A1..A12, B1..B12, ...


def _cycle(items: list[str], current: str, step: int) -> str:
    """Return the next/previous item from a list, with wraparound."""
    if not items:
        return current
    try:
        i = items.index(current)
    except ValueError:
        i = 0
    return items[(i + step) % len(items)]


def _delete_well_from_plate(plate: dict, well: str) -> None:
    """Remove a well from all per-well containers on a plate (in-place)."""

    per_well_keys = ["name", "raw_data", "processed_data", "growth_stats"]
    for k in per_well_keys:
        d = plate.get(k)
        d.pop(well, None)


def _get_selected_points(event) -> tuple[np.ndarray, np.ndarray]:
    """Extract selected x/y arrays from a Plotly selection event."""
    if event is None:
        return np.array([]), np.array([])

    sel = (
        event.get("selection")
        if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    if not sel:
        return np.array([]), np.array([])

    points = (
        sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    )
    if not points:
        return np.array([]), np.array([])

    xs = np.asarray([float(p["x"]) for p in points])
    ys = np.asarray([float(p["y"]) for p in points])
    return xs, ys


def update_growth_stats_from_lasso(
    plates: dict, pid: str, well: str, chart_key: str
) -> None:
    """
    Update growth stats based on lasso-selected points.

    For Sliding Window: Fit y = m x + b to selected points
    For Model Fitting: Refit the selected model to selected points only
    """
    xs, ys = _get_selected_points(st.session_state.get(chart_key))
    if xs.size < 2:
        return

    plate = plates.get(pid, {})
    gs = plates.setdefault(pid, {}).setdefault("growth_stats", {}).setdefault(well, {})

    # Store a timestamp to force UI update
    import time
    gs["_lasso_update_time"] = time.time()

    # Check which method was used
    fit_method = gs.get("fit_method", "Sliding Window")
    is_model_fit = fit_method and "Model Fitting" in str(fit_method)

    if is_model_fit:
        # Extract model type from fit_method string
        model_type = "logistic"  # default
        if "(" in fit_method and ")" in fit_method:
            model_type = fit_method.split("(")[1].split(")")[0]

        # Get all data points for the well
        processed = plate.get("processed_data", {}).get(well)
        if processed is None or processed.empty:
            return

        all_t = processed["Time"].to_numpy(float)
        all_y = processed["baseline_corrected"].to_numpy(float)

        # Filter to only the selected time range (with some tolerance)
        selected_t_min, selected_t_max = xs.min(), xs.max()
        time_tolerance = 0.1  # Allow 0.1 hour tolerance for point matching
        mask = (all_t >= selected_t_min - time_tolerance) & (all_t <= selected_t_max + time_tolerance)

        refit_t = all_t[mask]
        refit_y = all_y[mask]

        if refit_t.size < 5:
            # Not enough points for model fitting, fall back to linear fit of selected points
            # But preserve phase boundaries from original fit - don't set them to selection bounds
            m, b = np.polyfit(xs, ys, deg=1)
            t_mu = float(xs.mean())
            y_mu = float(m * t_mu + b)

            gs["Maximum U"] = float(m)
            gs["t_mu"] = t_mu
            gs["y_mu"] = y_mu
            gs["t_window_start"] = float(xs.min())
            gs["t_window_end"] = float(xs.max())
            # For Maximum OD600, use max from entire curve, not just selected points
            gs["Maximum OD600"] = float(all_y.max())
            # Preserve original phase boundaries - don't overwrite with selection bounds
            # gs["lag_phase_end"] and gs["exponential_phase_end"] are left unchanged
            # Store the selected time range even for linear fallback
            gs["lasso_t_min"] = float(selected_t_min)
            gs["lasso_t_max"] = float(selected_t_max)
            return

        # Get plate params for quality thresholds
        params = plate.get("params", {})

        # Refit the model to selected points only
        try:
            # Fit model to selected points
            fit_result = fit_growth_model(refit_t, refit_y, model_type=model_type)

            if fit_result is None:
                raise ValueError("Model fit failed")

            # Extract growth descriptors using the FULL time range for phase boundary calculation
            # This ensures phase boundaries can extend beyond the selected region
            descriptors = extract_growth_descriptors_from_model(
                fit_result,
                all_t,  # Use full time range, not just selected range
                frac_peak=float(params.get("lag_frac", 0.20))
            )

            # Update growth stats with refit results
            gs["Maximum U"] = descriptors["Maximum U"]
            gs["t_mu"] = descriptors["t_mu"]
            gs["y_mu"] = descriptors["y_mu"]
            gs["t_window_start"] = descriptors["t_window_start"]
            gs["t_window_end"] = descriptors["t_window_end"]
            gs["lag_phase_end"] = descriptors["lag_phase_end"]
            gs["exponential_phase_end"] = descriptors["exponential_phase_end"]
            gs["Maximum OD600"] = descriptors["Maximum OD600"]
            # Store the selected time range for plotting the refitted model curve
            gs["lasso_t_min"] = float(selected_t_min)
            gs["lasso_t_max"] = float(selected_t_max)
            # Keep the same fit_method

        except Exception:
            # If model fitting fails, fall back to linear fit
            # But preserve phase boundaries from original fit - don't set them to selection bounds
            m, b = np.polyfit(xs, ys, deg=1)
            t_mu = float(xs.mean())
            y_mu = float(m * t_mu + b)

            gs["Maximum U"] = float(m)
            gs["t_mu"] = t_mu
            gs["y_mu"] = y_mu
            gs["t_window_start"] = float(xs.min())
            gs["t_window_end"] = float(xs.max())
            # For Maximum OD600, use max from entire curve, not just selected points
            gs["Maximum OD600"] = float(all_y.max())
            # Preserve original phase boundaries - don't overwrite with selection bounds
            # gs["lag_phase_end"] and gs["exponential_phase_end"] are left unchanged
            # Store the selected time range even for linear fallback
            gs["lasso_t_min"] = float(selected_t_min)
            gs["lasso_t_max"] = float(selected_t_max)
    else:
        # Sliding Window method: use linear fit
        m, b = np.polyfit(xs, ys, deg=1)
        t_mu = float(xs.mean())
        y_mu = float(m * t_mu + b)

        gs["Maximum U"] = float(m)
        gs["t_mu"] = t_mu
        gs["y_mu"] = y_mu
        gs["t_window_start"] = float(xs.min())
        gs["t_window_end"] = float(xs.max())
        # Note: Maximum OD600, lag_phase_end, and exponential_phase_end are preserved
        # from the original analysis as they are not recalculated from lasso selection
        # in sliding window mode (only the growth rate window is updated)


# ---------------- Data helpers ----------------
def _sg_params_for_plate(plates: dict, plate_id: str) -> tuple[int, int, int]:
    """Return Savitzky-Golay and window parameters for a plate."""
    params = (plates.get(plate_id, {}) or {}).get("params") or {}
    return (
        int(params.get("sg_window", 11)),
        int(params.get("sg_poly", 2)),
        int(params.get("window_points", 15)),
    )


def analyse_well(record: dict, well: str) -> dict:
    """Recompute a single well fit from uploads and params."""
    u = (record or {}).get("uploads") or {}
    p = (record or {}).get("params") or {}

    well = str(well).upper()

    _, name_map = _plate_name_map(u["plate_bytes"])
    df = _read_table(u["data_bytes"], p["read_interval_min"])

    long = df.melt(id_vars="Time", var_name="well", value_name="value")
    long["well"] = long["well"].astype(str).str.upper()
    long["name"] = long["well"].map(name_map).fillna("False")
    long = long[long["name"] != "False"].copy()

    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    clip = p.get("clip_time_series", False)
    if clip:
        a, b = clip
        long = long.query("@a <= Time <= @b").copy()

    rm = p.get("remove_wells", False)
    if rm:
        long = long[~long["well"].isin([w.upper() for w in rm])].copy()

    # If the requested well was removed or doesn't exist after filtering:
    if well not in set(long["well"].unique()):
        return BAD_FIT.copy()

    long["od_1cm"] = long["value"] / float(p["pathlength_cm_"])

    baseline = pd.DataFrame()
    if p.get("blank", True):
        baseline = (
            long.query("name == 'BLANK'")
            .groupby("Time", as_index=True)["od_1cm"]
            .mean()
            .to_frame()
        )
        long = long.query("name != 'BLANK'").copy()

    if not baseline.empty:
        base = baseline["od_1cm"].to_dict()
        long["baseline_corrected"] = long["od_1cm"] - long["Time"].map(base).fillna(0.0)
    else:
        long["baseline_corrected"] = long["od_1cm"]

    g = long[long["well"] == well].copy()
    processed = g[["Time", "baseline_corrected"]].reset_index(drop=True)

    try:
        # Check which method to use
        growth_method = p.get("growth_method", "Sliding Window")
        if growth_method == "Model Fitting":
            fit = calculate_growth_descriptors_model_based(
                processed["Time"].to_numpy(float),
                processed["baseline_corrected"].to_numpy(float),
                model_type=p.get("model_type", "logistic"),
                lag_frac=float(p.get("lag_frac", 0.20)),
                min_data_points=int(p.get("min_data_points", 5)),
                min_signal_to_noise=float(p.get("min_signal_to_noise", 5.0)),
            )
        else:
            fit = calculate_growth_descriptors(
                processed["Time"].to_numpy(float),
                processed["baseline_corrected"].to_numpy(float),
                int(p["window_points"]),
                sg_window=int(p.get("sg_window", 11)),
                sg_poly=int(p.get("sg_poly", 2)),
                lag_frac=float(p.get("lag_frac", 0.20)),
                min_data_points=int(p.get("min_data_points", 5)),
                min_signal_to_noise=float(p.get("min_signal_to_noise", 5.0)),
            )
    except Exception:
        fit = BAD_FIT.copy()

    return fit


def _format_growth_stats_table(gs: dict) -> pd.DataFrame:
    """Format growth stats into a displayable table."""
    if not gs or is_bad_fit(gs):
        return pd.DataFrame({"Metric": ["No growth detected"], "Value": ["--"]})

    # Define metrics to display with nice labels and formatting
    metrics = [
        ("fit_method", "Fit Method", lambda x: str(x) if x else "Sliding Window"),
        ("Maximum OD600", "Maximum OD600", lambda x: f"{float(x):.4f}" if pd.notna(x) else "--"),
        ("Maximum U", "Maximum Growth Rate (1/h)", lambda x: f"{float(x):.4f}" if pd.notna(x) else "--"),
        ("t_mu", "Time at Max Growth (h)", lambda x: f"{float(x):.2f}" if pd.notna(x) else "--"),
        ("y_mu", "OD600 at Max Growth", lambda x: f"{float(x):.4f}" if pd.notna(x) else "--"),
        ("lag_phase_end", "Lag Phase End (h)", lambda x: f"{float(x):.2f}" if pd.notna(x) else "--"),
        ("exponential_phase_end", "Exponential Phase End (h)", lambda x: f"{float(x):.2f}" if pd.notna(x) else "--"),
        ("t_window_start", "Analysis Window Start (h)", lambda x: f"{float(x):.2f}" if pd.notna(x) else "--"),
        ("t_window_end", "Analysis Window End (h)", lambda x: f"{float(x):.2f}" if pd.notna(x) else "--"),
    ]

    rows = []
    for key, label, formatter in metrics:
        value = gs.get(key)
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
        st.warning(f"No data for {well}")
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
        lag0 = growth_stats.get("lag_phase_end")
        exp0 = growth_stats.get("exponential_phase_end")
        lag0 = float(lag0) if pd.notna(lag0) else t_min
        exp0 = float(exp0) if pd.notna(exp0) else t_min
        st.session_state[ss_key] = (lag0, exp0)

        st.session_state[maxod_key] = float(growth_stats.get("Maximum OD600", 0.0))

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
            st.session_state[ss_key],
            step=step,
            key=ss_key,
        )

    with c2:
        max_od = st.slider(
            "Set maximum OD600",
            0.0,
            max(processed["baseline_corrected"]),
            growth_stats.get("Maximum OD600", 0.0),
            step=max(processed["baseline_corrected"]) / 120,
            key=maxod_key,
        )

    # Persist boundaries unless we're deleting
    growth_stats["lag_phase_end"] = float(lag_end)
    growth_stats["exponential_phase_end"] = float(exp_end)
    growth_stats["Maximum OD600"] = float(max_od)

    c1, c2, c3 = st.columns(3)

    def _on_no_growth():
        """Mark the well as no-growth and reset widgets."""
        growth_stats.update(BAD_FIT.copy())
        _sync_widgets_from_growth_stats()

    def _on_reanalyse():
        """Re-run analysis for the well and refresh widgets."""
        plate["growth_stats"][well] = analyse_well(plate, well)
        # refresh local ref + sync widget state
        growth_stats.update(plate["growth_stats"][well])
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
        no_growth = st.button(
            "No Growth",
            width='stretch',
            type="primary",
            key=f"nogrowth__{key}",
            on_click=_on_no_growth,  # <-- changed
        )

    with c2:
        reanalyse_well = st.button(
            "Re-analyse",
            type="primary",
            width='stretch',
            on_click=_on_reanalyse,  # <-- changed
        )

    with c3:
        delete_well = st.button(
            "Exclude from analysis",
            width='stretch',
            type="tertiary",
            key=f"deletewell__{key}",
            on_click=_on_delete,  # <-- changed
        )

    st.write("")  # just adds some space

    return float(lag_end), float(exp_end), False


# ---------------- Window plot cache ----------------
@st.cache_data(show_spinner=False)
def _cached_window_single(processed_data: dict, well: str):
    """Cache the main well plot to avoid recomputation on reruns."""
    return plot_window_single(processed_data, well)


@st.cache_data(show_spinner=False)
def _cached_model_fit_single(processed_data: dict, growth_stats: dict, well: str, version_key: str = ""):
    """Cache the model fit plot to avoid recomputation on reruns. version_key busts cache when growth stats change."""
    return plot_model_fit_single(processed_data, growth_stats, well)


@st.fragment
def ui_window_fits_well_editor(plates: dict):
    """Render the well editor UI for interactive window fit adjustments."""
    plate_ids = sorted(plates)

    st.session_state.setdefault("winfit_plate", plate_ids[0])

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
    current_well = st.session_state.get("winfit_well", wells[0])
    if current_well not in wells:
        current_well = wells[0]
        st.session_state["winfit_well"] = current_well

    st.session_state.setdefault("winfit_well", wells[0])

    def _move_well(step: int):
        """Move the active well forward/backward."""
        st.session_state["winfit_well"] = _cycle(
            wells, st.session_state.get("winfit_well", wells[0]), step
        )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            plate_id = st.selectbox("Plate", plate_ids, key="winfit_plate")
            prev, mid, next_ = st.columns([2, 4, 2], vertical_alignment="bottom")
            with prev:
                st.button(
                    "",
                    width='stretch',
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
                    width='stretch',
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
    gs = growth_stats.get(well) or {}

    # Display growth stats table
    st.subheader(f"Growth Statistics for Well {well}")
    stats_df = _format_growth_stats_table(gs)
    # Use a key based on growth stats values to force update when they change
    # Include all key metrics that change during lasso selection
    table_key = (
        f"stats_table_{plate_id}_{well}_"
        f"{gs.get('Maximum U', 0)}_"
        f"{gs.get('Maximum OD600', 0)}_"
        f"{gs.get('lag_phase_end', 0)}_"
        f"{gs.get('exponential_phase_end', 0)}_"
        f"{gs.get('_lasso_update_time', '')}"
    )
    st.dataframe(stats_df, width="stretch", hide_index=True, key=table_key)

    st.divider()

    # Check which fitting method was used
    fit_method = gs.get("fit_method", "Sliding Window")
    is_model_fit = fit_method and "Model Fitting" in str(fit_method)

    chart_key = f"lasso_fit_{plate_id}_{well}"

    # Use appropriate plotting function based on fit method
    # For model fits, we need to regenerate the plot to show updated fits after lasso selection
    # Use a version key based on critical growth stats to bust cache when they change
    if is_model_fit:
        # Create a version string from the growth stats to trigger cache updates
        version_key = f"{gs.get('Maximum U', 0)}_{gs.get('t_mu', 0)}_{gs.get('y_mu', 0)}_{gs.get('lasso_t_min', '')}_{gs.get('lasso_t_max', '')}"
        fig_main = go.Figure(_cached_model_fit_single(processed, growth_stats, well, version_key))
    else:
        fig_main = go.Figure(_cached_window_single(processed, well))

    fig_main = _vlines(
        fig_main, processed, well, lag_end, exp_end, gs=gs
    )

    st.plotly_chart(
        fig_main,
        key=chart_key,
        selection_mode="lasso",
        on_select=lambda: update_growth_stats_from_lasso(
            plates, plate_id, well, chart_key
        ),
        width='stretch',
    )

    # Only show derivative plots for sliding window method
    if not is_model_fit:
        fig_d1 = plot_window_single_d1(
            plate, well, sg_window=sg_w, sg_poly=sg_p, frac_peak=0.20
        )
        fig_d2 = plot_window_single_d2(plate, well, sg_window=sg_w, sg_poly=sg_p)
        st.plotly_chart(fig_d1, width='stretch')
        st.plotly_chart(fig_d2, width='stretch')
    else:
        st.info(f"Growth descriptors calculated using {fit_method}. Derivative plots are only available for Sliding Window method.")
