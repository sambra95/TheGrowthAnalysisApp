"""Data loading, preprocessing, and growth-curve fitting utilities."""

import io

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

# Import all growth fitting functions from python_package
from src.python_package import (
    no_fit_dictionary,
    fit_growth_model,
    sliding_window_fit,
    detect_no_growth,
)

ROWS = "ABCDEFGH"
COLS = range(1, 13)
ALL_WELLS = [f"{r}{c}" for r in ROWS for c in COLS]


# ---------- small utilities ----------
def _as_float(x):
    """Convert array-like input to a float NumPy array."""
    return np.asarray(x, dtype=float)


@st.cache_data
def smooth(y, window=21, poly=1, passes=3):
    """Smooth a series with Savitzky-Golay filtering (odd window, multi-pass)."""
    y = _as_float(y)
    n = y.size
    if n < 7:
        return y
    w = int(window) | 1  # odd
    w = min(w, n if n % 2 else n - 1)
    p = min(int(poly), w - 1)
    for _ in range(int(passes)):
        y = savgol_filter(y, w, p, mode="interp")
    return y


def d1_model(t, A, r, t0):
    """Derivative of a logistic curve used for fitting growth rate peaks."""
    u = np.exp(-r * (t - t0))
    return A * (u / (1 + u) ** 2)


@st.cache_data
def fit_d1(t, dy):
    """Fit the derivative model to a gradient series; return (A, r, t0) or None."""
    t, dy = _as_float(t), np.maximum(_as_float(dy), 0.0)
    m = np.isfinite(t) & np.isfinite(dy)
    t, dy = t[m], dy[m]
    if t.size < 10 or np.ptp(t) <= 0 or dy.max(initial=0.0) <= 0:
        return None

    t0 = float(t[np.argmax(dy)])
    p0 = [float(4 * dy.max()), float(0.2 / max(np.ptp(t), 1e-9)), t0]
    bounds = ([0.0, 1e-6, float(t.min())], [np.inf, 10.0, float(t.max())])
    try:
        (A, r, t0), _ = curve_fit(d1_model, t, dy, p0=p0, bounds=bounds, maxfev=20000)
        return float(A), float(r), float(t0)
    except Exception:
        return None


@st.cache_data
def compute_first_derivative(t, y):
    """
    Compute the first derivative of a growth curve.

    Args:
        t: Time array
        y: OD600 values (baseline-corrected)

    Returns:
        Tuple of (t, dy) where dy is the first derivative dy/dt
    """
    t, y = _as_float(t), _as_float(y)
    dy = np.gradient(y, t)
    return t, dy


@st.cache_data
def compute_second_derivative(t, y):
    """
    Compute the second derivative of a growth curve.

    Args:
        t: Time array
        y: OD600 values (baseline-corrected)

    Returns:
        Tuple of (t, d2y) where d2y is the second derivative d²y/dt²
    """
    t, y = _as_float(t), _as_float(y)
    dy = np.gradient(y, t)
    d2y = np.gradient(dy, t)
    return t, d2y


def calculate_phase_ends(t, y_s, lag_frac=0.10, exp_frac=0.10):
    """Estimate lag and exponential phase end times from a smoothed curve.

    Args:
        t: Time array
        y_s: Smoothed OD600 values
        lag_frac: Fraction of peak growth rate for lag phase end detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection

    Returns:
        Tuple of (lag_end, exp_end) times
    """
    t, y_s = _as_float(t), _as_float(y_s)
    if t.size < 5 or np.ptp(t) <= 0:
        a = float(t[0]) if t.size else np.nan
        b = float(t[-1]) if t.size else np.nan
        return a, b

    p = fit_d1(t, np.gradient(y_s, t))
    if p is None:
        return float(t[0]), float(t[0])

    dy_fit = d1_model(t, *p)
    peak_i = int(np.nanargmax(dy_fit))
    peak_val = dy_fit[peak_i]

    lag_thr = float(lag_frac * peak_val)
    exp_thr = float(exp_frac * peak_val)

    lag_idx = np.where(dy_fit >= lag_thr)[0]
    exp_idx = np.where((dy_fit <= exp_thr) & (np.arange(t.size) > peak_i))[0]

    lag_end = float(t[lag_idx[0]]) if lag_idx.size else float(t[0])
    exp_end = float(t[exp_idx[0]]) if exp_idx.size else float(t[-1])
    return lag_end, max(exp_end, lag_end)


# ---------- I/O + shaping ----------
def _read_excel_bytes(b, **kw):
    """Read Excel bytes into a DataFrame."""
    return pd.read_excel(io.BytesIO(b), **kw)


def _plate_name_map(plate_bytes):
    """Return (plate_df, well_name_map) from a plate map Excel file."""
    plate = _read_excel_bytes(plate_bytes).fillna("False").set_index("rows")
    return plate, {f"{r}{c}": plate.loc[r, c] for r in ROWS for c in COLS}


def _read_table(data_bytes: bytes, time_unit: str = "hours") -> pd.DataFrame:
    """Read plate time series data (rows=timepoints, cols=wells) with Time in hours.

    Args:
        data_bytes: Excel file bytes
        time_unit: Unit of time in the data file ("seconds", "minutes", or "hours")

    Returns:
        DataFrame with Time column in hours and well columns

    Raises:
        ValueError: If no Time column is found in the data file
    """
    df = _read_excel_bytes(data_bytes, header=0)
    df = df.replace(",", ".", regex=True)

    if "Time" not in df.columns:
        raise ValueError(
            "Data file must contain a 'Time' column. "
            "Please add a Time column with numeric values."
        )

    t = pd.to_numeric(df["Time"], errors="coerce")
    df = df.drop(columns=["Time"])

    # Convert time to hours based on selected unit
    if time_unit == "seconds":
        t_hours = t / 3600.0
    elif time_unit == "minutes":
        t_hours = t / 60.0
    else:  # hours
        t_hours = t

    # keep only well-like columns (A1..H12) if extras exist
    valid_wells = {f"{r}{c}" for r in ROWS for c in COLS}
    well_cols = [c for c in df.columns if str(c).strip().upper() in valid_wells]
    df = df[well_cols].copy()
    df.columns = [str(c).strip().upper() for c in df.columns]

    df.insert(0, "Time", t_hours)
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _empty_plate():
    """Create an empty plate record structure."""
    return {"name": {}, "raw_data": {}, "processed_data": {}, "growth_stats": {}}


def load_plate(
    plates: dict, plate_id: str, *, data_bytes: bytes, plate_bytes: bytes, params: dict
):
    """Store uploads and params for a plate and return the record."""
    rec = plates.setdefault(plate_id, {})
    rec["uploads"] = {"data_bytes": data_bytes, "plate_bytes": plate_bytes}
    rec["params"] = params
    return rec


def analyse_plate(record: dict):
    """Process a plate record into cleaned, baseline-corrected per-well data."""
    u = (record or {}).get("uploads") or {}
    p = (record or {}).get("params") or {}

    plate_map, name_map = _plate_name_map(u["plate_bytes"])
    df = _read_table(u["data_bytes"], p.get("time_unit", "hours"))

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

    long["od_1cm"] = long["value"] / float(p["pathlength_cm_"])

    baseline = pd.DataFrame()
    if p.get("blank", True):
        blanks_long = long.query("name == 'BLANK'").copy()

        # one column per blank well (values are Mean), indexed by Time
        blanks_wide = blanks_long.pivot_table(
            index="Time",
            columns="well",
            values="od_1cm",
            aggfunc="mean",  # in case there are duplicates
        ).sort_index()

        if not blanks_wide.empty:
            # keep the existing mean column name for compatibility
            baseline = blanks_wide.copy()
            baseline["Mean"] = blanks_wide.mean(axis=1)

            # optional: put mean first (purely cosmetic)
            cols = ["Mean"] + [c for c in baseline.columns if c != "Mean"]
            baseline = baseline[cols]

        long = long.query("name != 'BLANK'").copy()

    if not baseline.empty:
        base = baseline["Mean"].to_dict()
        long["baseline_corrected"] = long["od_1cm"] - long["Time"].map(base).fillna(0.0)
    else:
        long["baseline_corrected"] = long["od_1cm"]

    plate = _empty_plate()
    plate["baseline"] = baseline
    plate["plate_map"] = plate_map

    # Get no-growth detection thresholds from params
    min_data_points = int(p.get("min_data_points", 5))
    min_signal_to_noise = float(p.get("min_signal_to_noise", 5.0))
    min_growth_rate = float(p.get("min_growth_rate", 0.001))

    for well, g in long.groupby("well", sort=False):
        processed = g[["Time", "baseline_corrected"]].reset_index(drop=True)

        try:
            # Choose method based on user selection
            # Uses python_package functions for growth statistics calculation
            growth_method = p.get("growth_method", "Sliding Window")
            t_arr = processed["Time"].to_numpy(float)
            y_arr = processed["baseline_corrected"].to_numpy(float)
            lag_frac = float(p.get("lag_cutoff", 0.15))
            exp_frac = float(p.get("exp_cutoff", 0.15))

            if growth_method == "Model Fitting":
                fit = fit_growth_model(
                    t_arr,
                    y_arr,
                    model_type=p.get("model_type", "logistic"),
                    lag_frac=lag_frac,
                    exp_frac=exp_frac,
                )
            else:
                fit = sliding_window_fit(
                    t_arr,
                    y_arr,
                    window_points=int(p["window_points"]),
                    sg_window=int(p.get("sg_window", 11)),
                    sg_poly=int(p.get("sg_poly", 1)),
                    lag_frac=lag_frac,
                    exp_frac=exp_frac,
                )
                fit["window_points"] = int(p["window_points"])

            # Check for no growth using consolidated detection function
            no_growth_result = detect_no_growth(
                t_arr,
                y_arr,
                growth_stats=fit,
                min_data_points=min_data_points,
                min_signal_to_noise=min_signal_to_noise,
                min_growth_rate=min_growth_rate,
            )
            if no_growth_result["is_no_growth"]:
                fit = no_fit_dictionary.copy()
                fit["no_growth_reason"] = no_growth_result["reason"]

        except Exception:
            fit = no_fit_dictionary.copy()

        plate["name"][well] = str(g["name"].iloc[0])
        plate["raw_data"][well] = g[["Time", "value", "od_1cm"]].reset_index(drop=True)
        plate["processed_data"][well] = processed
        plate["growth_stats"][well] = fit

    record.update(plate)
    return record


def compute_window_fits(
    plates,
    window_points=15,
    sg_window=11,
    sg_poly=2,
    lag_frac=0.10,
    exp_frac=0.10,
    min_data_points=5,
    min_signal_to_noise=5.0,
):
    """
    Recompute stats and write them back into plates[*]["processed_data"][well] in-place.
    Returns a stats dict-of-dicts: stats[plate_id][well] -> row dict for easy DF building.
    """
    stats = {}
    for plate_id, plate in plates.items():
        plate_stats = {}
        for well, wdict in plate["processed_data"].items():
            d = wdict["processed_values"]
            # Use python_package sliding_window_fit function
            fit = sliding_window_fit(
                d["Time"].to_numpy(float),
                d["baseline_corrected"].to_numpy(float),
                window_points=int(window_points),
                sg_window=int(sg_window),
                sg_poly=int(sg_poly),
                lag_frac=float(lag_frac),
                exp_frac=float(exp_frac),
            )

            wdict.update(
                {
                    "max_od": float(fit["max_od"]),
                    "specific_growth_rate": float(fit["specific_growth_rate"]),
                    "exp_phase_start": (
                        float(fit["exp_phase_start"])
                        if np.isfinite(fit["exp_phase_start"])
                        else np.nan
                    ),
                    "exp_phase_end": (
                        float(fit["exp_phase_end"])
                        if np.isfinite(fit["exp_phase_end"])
                        else np.nan
                    ),
                    "time_at_umax": (
                        float(fit["time_at_umax"])
                        if np.isfinite(fit["time_at_umax"])
                        else np.nan
                    ),
                    "od_at_umax": (
                        float(fit["od_at_umax"])
                        if np.isfinite(fit["od_at_umax"])
                        else np.nan
                    ),
                    "t_window_start": (
                        float(fit["t_window_start"])
                        if np.isfinite(fit["t_window_start"])
                        else np.nan
                    ),
                    "t_window_end": (
                        float(fit["t_window_end"])
                        if np.isfinite(fit["t_window_end"])
                        else np.nan
                    ),
                    "window_points": int(window_points),
                }
            )

            plate_stats[well] = {
                "Sample Name": plate["name"].get(well, ""),
                "max_od": wdict["max_od"],
                "specific_growth_rate": wdict["specific_growth_rate"],
            }
        stats[plate_id] = plate_stats
    return stats
