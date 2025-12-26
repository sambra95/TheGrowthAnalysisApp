import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions.data_processing import ALL_WELLS
from functions.plotting_functions import (
    _vlines,
    plot_window_single,
    plot_window_single_d1,
    plot_window_single_d2,
)

BAD_FIT = {
    "Maximum OD600": 0.0,
    "Maximum U": 0.0,
    "Lag Time (hours)": 0.0,
    "lag_phase_end": np.nan,
    "exponential_phase_end": np.nan,
    "t_mu": np.nan,
    "y_mu": np.nan,
    "b": np.nan,
    "t_peak": np.nan,
    "d1_fit": None,
}


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Selection + stats helpers ----------------
def _get_selected_points(event) -> tuple[np.ndarray, np.ndarray]:
    """Streamlit Plotly selection event -> arrays of selected x/y."""
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
    Read the plotly selection payload from st.session_state[chart_key],
    fit y = m x + b on selected points,
    and write results into plates[pid]["growth_stats"][well].
    """
    xs, ys = _get_selected_points(st.session_state.get(chart_key))
    if xs.size < 2:
        return

    m, b = np.polyfit(xs, ys, deg=1)
    t_mu = float(xs.mean())
    y_mu = float(m * t_mu + b)

    gs = plates.setdefault(pid, {}).setdefault("growth_stats", {}).setdefault(well, {})
    gs["Maximum U"] = float(m)
    gs["t_mu"] = t_mu
    gs["y_mu"] = y_mu
    gs["b"] = float(b)


# ---------------- Data helpers ----------------
def _sg_params_for_plate(plates: dict, plate_id: str) -> tuple[int, int, int]:
    params = (plates.get(plate_id, {}) or {}).get("params") or {}
    return (
        int(params.get("sg_window", 11)),
        int(params.get("sg_poly", 2)),
        int(params.get("window_points", 15)),
    )


def _phase_controls(plate: dict, well: str, *, key: str):
    """Range slider (lag_end, exp_end) + 'No Growth' button. Writes into plate['growth_stats'][well]."""
    processed = (plate.get("processed_data") or {}).get(well)
    if processed is None or processed.empty:
        st.warning(f"No data for {well}")
        return np.nan, np.nan, True

    t = processed["Time"]
    t_min, t_max = float(t.min()), float(t.max())
    step = float(max((t_max - t_min) / 200.0, 0.01))

    growth_stats = (plate.get("growth_stats") or {}).setdefault(well, {})
    ss_key = f"phase__{key}"

    if ss_key not in st.session_state:
        lag0 = growth_stats.get("lag_phase_end")
        exp0 = growth_stats.get("exponential_phase_end")
        lag0 = float(lag0) if pd.notna(lag0) else t_min
        exp0 = (
            float(exp0) if pd.notna(exp0) else min(t_min + 0.5 * (t_max - t_min), t_max)
        )
        st.session_state[ss_key] = (lag0, exp0)

    c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
    with c1:
        lag_end, exp_end = st.slider(
            "Phase boundaries (hours): Lag end → Exponential end",
            t_min,
            t_max,
            st.session_state[ss_key],
            step=step,
            key=ss_key,
        )
    with c2:
        no_growth = st.button(
            "No Growth",
            use_container_width=True,
            type="primary",
            key=f"nogrowth__{key}",
        )

    growth_stats["lag_phase_end"] = float(lag_end)
    growth_stats["exponential_phase_end"] = float(exp_end)

    if no_growth:
        growth_stats.update(BAD_FIT.copy())
        st.rerun()
        return np.nan, np.nan, True

    return float(lag_end), float(exp_end), False


# ---------------- Window plot cache ----------------
@st.cache_data(show_spinner=False)
def _cached_window_single(processed_data: dict, well: str):
    return plot_window_single(processed_data, well)


# ---------------- Fragments used by pages ----------------
@st.fragment
def ui_window_fits_well_editor(plates: dict, *, line_hours: float = 4.0):
    a, b = st.columns(2)
    plate_id = a.selectbox("Plate", sorted(plates), key="winfit_plate")
    well = b.selectbox("Well", ALL_WELLS, key="winfit_well")

    plate = plates[plate_id]
    key = f"{plate_id}_{well}"

    lag_end, exp_end, no_growth = _phase_controls(plate, well, key=key)
    if no_growth:
        return

    sg_w, sg_p, _ = _sg_params_for_plate(plates, plate_id)
    processed = plate.get("processed_data") or {}
    gs = (plate.get("growth_stats") or {}).get(well) or {}

    fig_d1 = plot_window_single_d1(
        plate, well, sg_window=sg_w, sg_poly=sg_p, frac_peak=0.20
    )
    fig_d2 = plot_window_single_d2(plate, well, sg_window=sg_w, sg_poly=sg_p)

    chart_key = f"lasso_fit_{plate_id}_{well}"
    fig_main = go.Figure(_cached_window_single(processed, well))  # IMPORTANT: copy!
    fig_main = _vlines(
        fig_main, processed, well, lag_end, exp_end, gs=gs, line_hours=line_hours
    )

    st.plotly_chart(
        fig_main,
        key=chart_key,
        selection_mode="lasso",
        on_select=lambda: update_growth_stats_from_lasso(
            plates, plate_id, well, chart_key
        ),
        use_container_width=True,
    )
    st.plotly_chart(fig_d1, use_container_width=True)
    st.plotly_chart(fig_d2, use_container_width=True)


# ---------------- Backwards-compatible alias ----------------
window_well_view = ui_window_fits_well_editor
