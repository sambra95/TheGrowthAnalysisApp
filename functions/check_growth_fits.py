"""Interactive well-by-well growth fit inspection helpers."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from growthcurves.inference import bad_fit_stats

from functions.fitting_pipeline import fit_growth_series
from functions.plotting_functions import is_bad_fit


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


def _match_selected_times(
    all_t: np.ndarray, selected_times: np.ndarray, *, time_tolerance: float = 0.01
) -> np.ndarray:
    """Return indices in all_t that match selected_times within tolerance."""
    if all_t.size == 0 or selected_times.size == 0:
        return np.array([], dtype=int)

    refit_indices = []
    seen = set()
    for sel_t in selected_times:
        matches = np.where(np.abs(all_t - sel_t) < time_tolerance)[0]
        if len(matches) == 0:
            continue
        idx = int(matches[0])
        if idx not in seen:
            refit_indices.append(idx)
            seen.add(idx)

    return np.asarray(refit_indices, dtype=int)


def _add_lasso_selected_points(
    fig: go.Figure,
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    selected_times: list[float] | None,
    *,
    scale: str,
    time_tolerance: float = 0.01,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Overlay lasso-selected points in red on an existing figure."""
    if not selected_times:
        return fig

    sel = np.asarray(selected_times, dtype=float)
    sel = sel[np.isfinite(sel)]
    if sel.size == 0:
        return fig

    refit_indices = _match_selected_times(t_arr, sel, time_tolerance=time_tolerance)
    if refit_indices.size == 0:
        return fig

    t_sel = t_arr[refit_indices]
    y_sel = y_arr[refit_indices]
    if scale == "log":
        valid = (y_sel > 0) & np.isfinite(y_sel)
        t_sel = t_sel[valid]
        y_sel = y_sel[valid]
        if t_sel.size == 0:
            return fig
        y_plot = np.log(y_sel)
    else:
        valid = np.isfinite(y_sel)
        t_sel = t_sel[valid]
        y_plot = y_sel[valid]
        if t_sel.size == 0:
            return fig

    fig.add_trace(
        go.Scatter(
            x=t_sel,
            y=y_plot,
            mode="markers",
            marker=dict(size=8, color="#ef5350", symbol="circle"),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    return fig


def _collect_lasso_series(
    processed: pd.DataFrame, selected_times: np.ndarray, *, time_tolerance: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, y) arrays for lasso-selected points from processed data."""
    if processed is None or processed.empty:
        return np.array([]), np.array([])

    all_t = processed["Time"].to_numpy(float)
    all_y = processed["baseline_corrected"].to_numpy(float)

    refit_indices = _match_selected_times(
        all_t, selected_times, time_tolerance=time_tolerance
    )
    if refit_indices.size < 2:
        return np.array([]), np.array([])

    refit_t = all_t[refit_indices]
    refit_y = all_y[refit_indices]

    sort_idx = np.argsort(refit_t)
    return refit_t[sort_idx], refit_y[sort_idx]


def _analyse_series_with_plate_params(
    t_arr: np.ndarray, y_arr: np.ndarray, params: dict
) -> tuple[dict, dict | None]:
    """Run the same analysis pipeline as initial plate analysis."""
    if t_arr.size < 2 or y_arr.size < 2:
        return bad_fit_stats(), None

    try:
        # Use unified fitting pipeline
        return fit_growth_series(t_arr, y_arr, params)
    except Exception:
        return bad_fit_stats(), None


# ---------------- Plot helpers ----------------
def _as_finite_float(value) -> float | None:
    """Return a finite float or None."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if np.isfinite(num) else None


def update_growth_stats_from_lasso(
    plates: dict, pid: str, well: str, chart_key: str
) -> None:
    """
    Update growth stats based on lasso-selected points.

    For Sliding Window: Re-run sliding window analysis on selected points only
    For Model Fitting: Refit the selected model to selected points only
    """
    xs, ys = _get_selected_points(st.session_state.get(chart_key))
    if xs.size < 2:
        return

    plate = plates.get(pid, {})
    gs = plates.setdefault(pid, {}).setdefault("growth_stats", {}).setdefault(well, {})
    fit_parameters = plates.setdefault(pid, {}).setdefault("fit_parameters", {})

    # Store a timestamp to force UI update
    import time

    gs["_lasso_update_time"] = time.time()

    # Store the actual selected time values for visualization (red vs grey points)
    # Sort by time to ensure consistent ordering
    sort_idx = np.argsort(xs)
    selected_times = xs[sort_idx]

    # Get all data points for the well (always stored internally in hours)
    processed = plate.get("processed_data", {}).get(well)
    if processed is None or processed.empty:
        return

    # Build lasso-selected series and store for visualization
    refit_t, refit_y = _collect_lasso_series(processed, selected_times)
    if refit_t.size < 2:
        return

    gs["_used_fit_times"] = refit_t.tolist()

    # Run the identical analysis pipeline as initial plate analysis
    params = plate.get("params", {})
    fit, fit_result = _analyse_series_with_plate_params(refit_t, refit_y, params)
    gs.update(fit)
    if fit_result is not None:
        fit_parameters[well] = fit_result
    else:
        fit_parameters.pop(well, None)


# ---------------- Data helpers ----------------
def _sg_params_for_plate(plates: dict, plate_id: str) -> tuple[int, int, int]:
    """Return Savitzky-Golay and window parameters for a plate."""
    params = (plates.get(plate_id, {}) or {}).get("params") or {}
    return (
        int(params.get("sg_window", 11)),
        int(params.get("sg_poly", 2)),
        int(params.get("window_points", 15)),
    )


def analyse_well(plate: dict, well: str) -> dict:
    """Recompute growth statistics for a single well using existing processed data."""
    p = (plate or {}).get("params") or {}
    processed_data = (plate or {}).get("processed_data") or {}
    fit_parameters = (plate or {}).setdefault("fit_parameters", {})

    well = str(well).upper()

    # Get the already-processed data for this well
    processed = processed_data.get(well)
    if processed is None or processed.empty:
        fit_parameters.pop(well, None)
        return bad_fit_stats()

    try:
        t_arr = processed["Time"].to_numpy(float)
        y_arr = processed["baseline_corrected"].to_numpy(float)
        fit, fit_result = _analyse_series_with_plate_params(t_arr, y_arr, p)
        if fit_result is not None:
            fit_parameters[well] = fit_result
        else:
            fit_parameters.pop(well, None)

    except Exception:
        fit = bad_fit_stats()
        fit_parameters.pop(well, None)

    return fit
