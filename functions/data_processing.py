"""Data loading, preprocessing, and growth-curve fitting utilities."""

import io

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

ROWS = "ABCDEFGH"
COLS = range(1, 13)
ALL_WELLS = [f"{r}{c}" for r in ROWS for c in COLS]


BAD_FIT = {
    "Maximum OD600": 0.0,
    "Maximum U": 0.0,
    "lag_phase_end": np.nan,
    "exponential_phase_end": np.nan,
    "t_mu": np.nan,
    "y_mu": np.nan,
    "t_window_start": np.nan,
    "t_window_end": np.nan,
    "fit_method": None,
    "rmse": np.nan,
}


# ---------- small utilities ----------
def _as_float(x):
    """Convert array-like input to a float NumPy array."""
    return np.asarray(x, dtype=float)


def calculate_rmse(y_observed, y_predicted):
    """
    Calculate Root Mean Square Error between observed and predicted values.

    Args:
        y_observed: Observed/actual values
        y_predicted: Predicted values from model

    Returns:
        RMSE value as float
    """
    y_observed = _as_float(y_observed)
    y_predicted = _as_float(y_predicted)

    # Filter out non-finite values
    mask = np.isfinite(y_observed) & np.isfinite(y_predicted)
    if mask.sum() == 0:
        return np.nan

    residuals = y_observed[mask] - y_predicted[mask]
    rmse = np.sqrt(np.mean(residuals ** 2))
    return float(rmse)


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


def calculate_phase_ends(t, y_s, frac_peak=0.10):
    """Estimate lag and exponential phase end times from a smoothed curve."""
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
    thr = float(frac_peak * dy_fit[peak_i])

    lag_idx = np.where(dy_fit >= thr)[0]
    exp_idx = np.where((dy_fit <= thr) & (np.arange(t.size) > peak_i))[0]

    lag_end = float(t[lag_idx[0]]) if lag_idx.size else float(t[0])
    exp_end = float(t[exp_idx[0]]) if exp_idx.size else float(t[-1])
    return lag_end, max(exp_end, lag_end)


# ---------- Growth curve models ----------
def logistic_model(t, A, mu, lag):
    """Logistic growth model: y(t) = A / (1 + exp(-mu * (t - lag)))"""
    return A / (1 + np.exp(-mu * (t - lag)))


def gompertz_model(t, A, mu, lag):
    """Gompertz growth model: y(t) = A * exp(-exp(-mu * (t - lag)))"""
    return A * np.exp(-np.exp(-mu * (t - lag)))


def richards_model(t, A, mu, lag, nu):
    """Richards growth model (generalized logistic with shape parameter nu)"""
    return A / (1 + nu * np.exp(-mu * (t - lag))) ** (1 / nu)


@st.cache_data
def fit_growth_model(t, y, model_type="logistic"):
    """
    Fit a parametric growth model to data.

    Args:
        t: Time array
        y: OD600 values (baseline-corrected)
        model_type: One of "logistic", "gompertz", "richards", "spline"

    Returns:
        Dictionary with fitted parameters and model info, or None if fit fails
    """
    t, y = _as_float(t), _as_float(y)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    if t.size < 10 or np.ptp(t) <= 0:
        return None

    if model_type == "spline":
        # Fit a smoothing spline
        try:
            # Use adaptive smoothing based on data size
            s = max(0.01 * t.size, 1.0)
            spline = UnivariateSpline(t, y, s=s, k=3)
            return {"type": "spline", "spline": spline, "t": t, "y": y}
        except Exception:
            return None

    # Parametric models
    A_init = float(np.nanmax(y))
    lag_init = float(t[np.argmax(np.gradient(y, t))])
    mu_init = 0.5

    if model_type == "logistic":
        model_func = logistic_model
        p0 = [A_init, mu_init, lag_init]
        bounds = ([0, 0, float(t.min())], [np.inf, 10, float(t.max())])
        param_names = ["A", "mu", "lag"]
    elif model_type == "gompertz":
        model_func = gompertz_model
        p0 = [A_init, mu_init, lag_init]
        bounds = ([0, 0, float(t.min())], [np.inf, 10, float(t.max())])
        param_names = ["A", "mu", "lag"]
    elif model_type == "richards":
        model_func = richards_model
        p0 = [A_init, mu_init, lag_init, 1.0]
        bounds = ([0, 0, float(t.min()), 0.01], [np.inf, 10, float(t.max()), 100])
        param_names = ["A", "mu", "lag", "nu"]
    else:
        return None

    try:
        params, _ = curve_fit(model_func, t, y, p0=p0, bounds=bounds, maxfev=20000)
        return {
            "type": model_type,
            "model": model_func,
            "params": {name: float(val) for name, val in zip(param_names, params)},
            "t": t,
            "y": y,
        }
    except Exception:
        return None


def extract_growth_descriptors_from_model(fit_result, t_data, frac_peak=0.10, y_data=None):
    """
    Extract growth descriptors from a fitted growth model.

    Args:
        fit_result: Dictionary returned by fit_growth_model
        t_data: Original time array for generating dense predictions
        frac_peak: Fraction of peak growth rate for phase boundary detection
        y_data: Optional observed y values for RMSE calculation (if None, uses fit_result['y'])

    Returns:
        Dictionary with growth descriptors (same format as calculate_growth_descriptors)
    """
    if fit_result is None:
        return BAD_FIT.copy()

    # Use the y_data from the fit if not provided separately
    if y_data is None:
        y_data = fit_result.get('y')

    t_dense = np.linspace(float(t_data.min()), float(t_data.max()), 500)

    # Generate predictions
    if fit_result["type"] == "spline":
        y_pred = fit_result["spline"](t_dense)
        dy_pred = fit_result["spline"].derivative()(t_dense)
    else:
        model_func = fit_result["model"]
        params = fit_result["params"]
        if fit_result["type"] == "richards":
            y_pred = model_func(t_dense, params["A"], params["mu"], params["lag"], params["nu"])
        else:
            y_pred = model_func(t_dense, params["A"], params["mu"], params["lag"])
        dy_pred = np.gradient(y_pred, t_dense)

    # Maximum OD600
    max_od_idx = np.nanargmax(y_pred)
    max_od = float(y_pred[max_od_idx])

    # Maximum growth rate (U) and its location
    max_u_idx = np.nanargmax(dy_pred)
    max_u = float(dy_pred[max_u_idx])
    t_mu = float(t_dense[max_u_idx])
    y_mu = float(y_pred[max_u_idx])

    if max_u <= 0:
        out = BAD_FIT.copy()
        out["Maximum OD600"] = max_od
        return out

    # Calculate phase boundaries using derivative threshold
    peak_dy = dy_pred[max_u_idx]
    threshold = frac_peak * peak_dy

    # Lag phase end: first time derivative exceeds threshold
    lag_idx = np.where(dy_pred >= threshold)[0]
    lag_end = float(t_dense[lag_idx[0]]) if lag_idx.size else float(t_dense[0])

    # Exponential phase end: first time after peak where derivative drops below threshold
    exp_idx = np.where((dy_pred <= threshold) & (np.arange(t_dense.size) > max_u_idx))[0]
    exp_end = float(t_dense[exp_idx[0]]) if exp_idx.size else float(t_dense[-1])

    # For model-based fits, window start/end represent the confidence region around max growth
    # Use a narrow window around the inflection point
    window_half = 0.1 * (t_dense[-1] - t_dense[0])  # 10% of time range
    t_window_start = max(float(t_mu - window_half), float(t_dense[0]))
    t_window_end = min(float(t_mu + window_half), float(t_dense[-1]))

    # Calculate RMSE using the original data points that were fitted
    rmse = np.nan
    if y_data is not None and len(y_data) > 0:
        t_fit = fit_result.get('t', t_data)
        # Generate predictions at the fitted time points
        if fit_result["type"] == "spline":
            y_pred_fit = fit_result["spline"](t_fit)
        else:
            model_func = fit_result["model"]
            params = fit_result["params"]
            if fit_result["type"] == "richards":
                y_pred_fit = model_func(t_fit, params["A"], params["mu"], params["lag"], params["nu"])
            else:
                y_pred_fit = model_func(t_fit, params["A"], params["mu"], params["lag"])
        rmse = calculate_rmse(y_data, y_pred_fit)

    return {
        "Maximum OD600": max_od,
        "Maximum U": max_u,
        "lag_phase_end": lag_end,
        "exponential_phase_end": max(exp_end, lag_end),
        "t_mu": t_mu,
        "y_mu": y_mu,
        "t_window_start": t_window_start,
        "t_window_end": t_window_end,
        "fit_method": f"Model Fitting ({fit_result['type']})",
        "rmse": rmse,
    }


def calculate_growth_descriptors_model_based(
    t,
    y,
    model_type="logistic",
    *,
    lag_frac=0.10,
    min_data_points=5,
    min_signal_to_noise=5.0,
):
    """
    Calculate growth descriptors by fitting a parametric growth model.

    Args:
        t: Time array
        y: OD600 values (baseline-corrected)
        model_type: One of "logistic", "gompertz", "richards", "spline"
        lag_frac: Fraction of peak for phase boundary detection
        min_data_points: Minimum number of data points required for valid fit
        min_signal_to_noise: Minimum ratio of max/min signal for valid fit

    Returns:
        Dictionary with growth descriptors
    """
    t, y = _as_float(t), _as_float(y)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]

    # Check data quality
    if t.size < max(int(min_data_points), 10) or np.ptp(t) <= 0:
        return BAD_FIT.copy()

    if abs(y.max() / max(abs(y.min()), 1e-12)) <= float(min_signal_to_noise):
        return BAD_FIT.copy()

    # Fit the growth model
    fit_result = fit_growth_model(t, y, model_type=model_type)

    # Extract growth descriptors from the fitted model
    return extract_growth_descriptors_from_model(fit_result, t, frac_peak=lag_frac)


def calculate_growth_descriptors(
    t,
    y,
    w=15,
    *,
    sg_window=11,
    sg_poly=1,
    lag_frac=0.10,
    min_data_points=5,
    min_signal_to_noise=5.0,
):
    """Fit a sliding-window line to find max growth slope and phase boundaries.

    Args:
        t: Time array
        y: OD600 values (baseline-corrected)
        w: Window size for sliding window fit
        sg_window: Savitzky-Golay filter window size
        sg_poly: Savitzky-Golay filter polynomial order
        lag_frac: Fraction of peak for phase boundary detection
        min_data_points: Minimum number of data points required for valid fit
        min_signal_to_noise: Minimum ratio of max/min signal for valid fit
    """
    t, y = _as_float(t), _as_float(y)
    w = int(w)

    # Return no fit if insufficient data.
    y_s = smooth(y, sg_window, sg_poly)
    m = np.isfinite(t) & np.isfinite(y_s)
    t, y_s = t[m], y_s[m]
    if t.size < max(int(min_data_points), w) or np.ptp(t) <= 0:
        return BAD_FIT.copy()

    if abs(y_s.max() / max(abs(y_s.min()), 1e-12)) <= float(min_signal_to_noise):
        return BAD_FIT.copy()

    # Take max OD measurement from non-smoothed line.
    peak_i = int(np.nanargmax(y))
    A = float(y[peak_i])

    # calculate the growth rate (use raw data for fit, like manual lasso selection)
    w = min(w, t.size)
    best_m, best = -np.inf, (np.nan, np.nan, np.nan, np.nan)  # (t_mu, y_mu, t_start, t_end)
    for i in range(t.size - w + 1):
        tw, yw = t[i : i + w], y[i : i + w]  # use raw y, not smoothed y_s
        if np.ptp(tw) <= 0:
            continue
        m_i, b_i = np.polyfit(tw, yw, 1)
        t_mu = float(tw.mean())
        if m_i > best_m:
            best_m = float(m_i)
            best = (t_mu, float(best_m * t_mu + b_i), float(tw[0]), float(tw[-1]))

    t_mu, y_mu, t_window_start, t_window_end = best
    if not np.isfinite(best_m) or best_m <= 0:
        out = BAD_FIT.copy()
        out.update({"Maximum OD600": A})
        return out

    # calculate exponential phase start and end
    lag_end, exp_end = calculate_phase_ends(t, y_s, frac_peak=lag_frac)

    out = {
        "Maximum OD600": A,
        "Maximum U": float(best_m),
        "lag_phase_end": float(lag_end),
        "exponential_phase_end": float(exp_end),
        "t_mu": float(t_mu),
        "y_mu": float(y_mu),
        "t_window_start": float(t_window_start),
        "t_window_end": float(t_window_end),
        "fit_method": "Sliding Window",
        "rmse": np.nan,  # RMSE not applicable for sliding window method
    }

    return out


# ---------- I/O + shaping ----------
def _read_excel_bytes(b, **kw):
    """Read Excel bytes into a DataFrame."""
    return pd.read_excel(io.BytesIO(b), **kw)


def _plate_name_map(plate_bytes):
    """Return (plate_df, well_name_map) from a plate map Excel file."""
    plate = _read_excel_bytes(plate_bytes).fillna("False").set_index("rows")
    return plate, {f"{r}{c}": plate.loc[r, c] for r in ROWS for c in COLS}


def _read_table(data_bytes: bytes, read_interval_min: int) -> pd.DataFrame:
    """Read plate time series data (rows=timepoints, cols=wells) with Time in hours."""
    df = _read_excel_bytes(data_bytes, header=0)
    df = df.replace(",", ".", regex=True)

    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce")
        df = df.drop(columns=["Time"])
    else:
        # assume each row is a timepoint, evenly spaced
        t = pd.Series(np.arange(len(df)) * int(read_interval_min))

    # keep only well-like columns (A1..H12) if extras exist
    valid_wells = {f"{r}{c}" for r in ROWS for c in COLS}
    well_cols = [c for c in df.columns if str(c).strip().upper() in valid_wells]
    df = df[well_cols].copy()
    df.columns = [str(c).strip().upper() for c in df.columns]

    df.insert(0, "Time", t / 60.0)  # hours
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

    for well, g in long.groupby("well", sort=False):
        processed = g[["Time", "baseline_corrected"]].reset_index(drop=True)

        try:
            # Choose method based on user selection
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
    lag_frac=0.20,
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
            fit = calculate_growth_descriptors(
                d["Time"].to_numpy(float),
                d["baseline_corrected"].to_numpy(float),
                int(window_points),
                sg_window=int(sg_window),
                sg_poly=int(sg_poly),
                lag_frac=float(lag_frac),
                min_data_points=int(min_data_points),
                min_signal_to_noise=float(min_signal_to_noise),
            )

            wdict.update(
                {
                    "Maximum OD600": float(fit["Maximum OD600"]),
                    "Maximum U": float(fit["Maximum U"]),
                    "lag_phase_end": (
                        float(fit["lag_phase_end"])
                        if np.isfinite(fit["lag_phase_end"])
                        else np.nan
                    ),
                    "exponential_phase_end": (
                        float(fit["exponential_phase_end"])
                        if np.isfinite(fit["exponential_phase_end"])
                        else np.nan
                    ),
                    "t_mu": float(fit["t_mu"]) if np.isfinite(fit["t_mu"]) else np.nan,
                    "y_mu": float(fit["y_mu"]) if np.isfinite(fit["y_mu"]) else np.nan,
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
                }
            )

            plate_stats[well] = {
                "Sample Name": plate["name"].get(well, ""),
                "Maximum OD600": wdict["Maximum OD600"],
                "Maximum U": wdict["Maximum U"],
            }
        stats[plate_id] = plate_stats
    return stats
