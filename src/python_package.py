"""Growth curve fitting and analysis functions.

This module provides functions to fit growth models (Richards, Logistic, Gompertz)
and calculate growth statistics from time series data.

All models operate in linear OD space (not log-transformed).
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


# -----------------------------------------------------------------------------
# Growth Models (operate on linear OD data)
# -----------------------------------------------------------------------------


def logistic_model(t, K, y0, r, t0):
    """
    Logistic growth model in linear OD space with baseline offset.

    N(t) = y0 + (K - y0) / [1 + exp(-r * (t - t0))]

    Parameters:
        t: Time array
        K: Carrying capacity (maximum OD)
        y0: Baseline OD (minimum value at t=0)
        r: Growth rate constant (h^-1), equals mu_max
        t0: Inflection time (time at which N = (K + y0)/2)

    Returns:
        OD values at each time point

    Note:
        mu_max = r for the logistic model
    """
    return y0 + (K - y0) / (1 + np.exp(-r * (t - t0)))


def gompertz_model(t, K, y0, mu_max, lam):
    """
    Modified Gompertz growth model in linear OD space with baseline offset.

    N(t) = y0 + (K - y0) * exp{-exp[(mu_max * e / (K - y0)) * (lam - t) + 1]}

    Parameters:
        t: Time array
        K: Carrying capacity (maximum OD)
        y0: Baseline OD (minimum value)
        mu_max: Maximum specific growth rate (h^-1)
        lam: Lag time (hours)

    Returns:
        OD values at each time point

    Note:
        mu_max is directly a fitted parameter
    """
    e = np.e
    A = K - y0  # Amplitude
    return y0 + A * np.exp(-np.exp((mu_max * e / A) * (lam - t) + 1))


def richards_model(t, K, y0, r, t0, nu):
    """
    Richards growth model in linear OD space with baseline offset.

    N(t) = y0 + (K - y0) * [1 + nu * exp(-r * (t - t0))]^(-1/nu)

    Parameters:
        t: Time array
        K: Carrying capacity (maximum OD)
        y0: Baseline OD (minimum value)
        r: Growth rate constant (h^-1)
        t0: Inflection time
        nu: Shape parameter (nu=1 gives logistic, nu->0 gives Gompertz)

    Returns:
        OD values at each time point

    Note:
        mu_max = r / (nu + 1)^(1/nu) for the Richards model
    """
    return y0 + (K - y0) * (1 + nu * np.exp(-r * (t - t0))) ** (-1 / nu)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _validate_data(t, y, min_points=10):
    """
    Validate and clean input data.

    Returns:
        Tuple of (t, y) arrays with finite values only, or (None, None) if invalid.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y = t[mask], y[mask]

    if len(t) < min_points or np.ptp(t) <= 0:
        return None, None

    return t, y


def _compute_rmse(y_observed, y_predicted):
    """Calculate root mean square error between observed and predicted values."""
    mask = np.isfinite(y_observed) & np.isfinite(y_predicted)
    if mask.sum() == 0:
        return np.nan
    residuals = y_observed[mask] - y_predicted[mask]
    return float(np.sqrt(np.mean(residuals**2)))


def _calculate_specific_growth_rate(t, y_fit):
    """
    Calculate the maximum specific growth rate from a fitted curve.

    The specific growth rate is μ = (1/N) * dN/dt = d(ln(N))/dt

    Parameters:
        t: Time array
        y_fit: Fitted OD values

    Returns:
        Maximum specific growth rate (time^-1)
    """
    # Use dense time points for accurate derivative
    t_dense = np.linspace(t.min(), t.max(), 1000)
    y_interp = np.interp(t_dense, t, y_fit)

    # Calculate specific growth rate: μ = (1/N) * dN/dt
    dN_dt = np.gradient(y_interp, t_dense)

    # Avoid division by very small values
    y_safe = np.maximum(y_interp, 1e-10)
    mu = dN_dt / y_safe

    # Return maximum specific growth rate
    return float(np.max(mu))


def _smooth(y, window=11, poly=1, passes=2):
    """Apply Savitzky-Golay smoothing filter."""
    n = len(y)
    if n < 7:
        return y
    w = int(window) | 1  # Ensure odd
    w = min(w, n if n % 2 else n - 1)
    p = min(int(poly), w - 1)
    for _ in range(passes):
        y = savgol_filter(y, w, p, mode="interp")
    return y


# -----------------------------------------------------------------------------
# Derivative Functions
# -----------------------------------------------------------------------------


def first_derivative(t, y):
    """
    Calculate the first derivative (dN/dt) of a curve.

    Parameters:
        t: Time array
        y: Values array (e.g., OD or fitted values)

    Returns:
        Array of first derivative values (same length as input)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.gradient(y, t)


def second_derivative(t, y):
    """
    Calculate the second derivative (d²N/dt²) of a curve.

    Parameters:
        t: Time array
        y: Values array (e.g., OD or fitted values)

    Returns:
        Array of second derivative values (same length as input)
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dy_dt = np.gradient(y, t)
    return np.gradient(dy_dt, t)


# -----------------------------------------------------------------------------
# Model Fitting Functions
# -----------------------------------------------------------------------------


def fit_logistic(t, y):
    """
    Fit logistic model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params', 'y_fit', 't' or None if fitting fails.
    """
    t, y = _validate_data(t, y)
    if t is None:
        return None

    # Initial estimates
    K_init = np.max(y)
    y0_init = np.min(y)  # Baseline OD
    # Estimate t0 as time of maximum growth rate (inflection point)
    dy = np.gradient(y, t)
    t0_init = t[np.argmax(dy)]
    r_init = 0.01  # Initial growth rate guess

    p0 = [K_init, y0_init, r_init, t0_init]
    bounds = ([y0_init * 0.5, 0, 0.0001, t.min()], [np.inf, y0_init * 2, 10, t.max()])

    try:
        params, _ = curve_fit(logistic_model, t, y, p0=p0, bounds=bounds, maxfev=20000)
        y_fit = logistic_model(t, *params)
        # Calculate true specific growth rate from fitted curve
        mu_max = _calculate_specific_growth_rate(t, y_fit)
        return {
            "params": {
                "K": params[0],
                "y0": params[1],
                "r": params[2],
                "t0": params[3],
            },
            "y_fit": y_fit,
            "y": y,
            "t": t,
            "model_type": "logistic",
            "mu_max": mu_max,
        }
    except Exception:
        return None


def fit_gompertz(t, y):
    """
    Fit modified Gompertz model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params', 'y_fit', 't' or None if fitting fails.
    """
    t, y = _validate_data(t, y)
    if t is None:
        return None

    # Initial estimates
    K_init = np.max(y)
    y0_init = np.min(y)  # Baseline OD
    # Estimate lag time as time when growth first accelerates
    dy = np.gradient(y, t)
    threshold = 0.1 * np.max(dy)
    lag_idx = np.where(dy > threshold)[0]
    lam_init = t[lag_idx[0]] if len(lag_idx) > 0 else t[0]
    mu_max_init = 0.01  # Initial growth rate guess

    p0 = [K_init, y0_init, mu_max_init, lam_init]
    bounds = ([y0_init * 0.5, 0, 0.0001, 0], [np.inf, y0_init * 2, 10, t.max()])

    try:
        params, _ = curve_fit(gompertz_model, t, y, p0=p0, bounds=bounds, maxfev=20000)
        y_fit = gompertz_model(t, *params)
        # Calculate true specific growth rate from fitted curve
        mu_max = _calculate_specific_growth_rate(t, y_fit)
        return {
            "params": {
                "K": params[0],
                "y0": params[1],
                "mu_max_param": params[2],
                "lam": params[3],
            },
            "y_fit": y_fit,
            "y": y,
            "t": t,
            "model_type": "gompertz",
            "mu_max": mu_max,
        }
    except Exception:
        return None


def fit_richards(t, y):
    """
    Fit Richards model to growth data.

    Parameters:
        t: Time array (hours)
        y: OD values

    Returns:
        Dict with 'params', 'y_fit', 't' or None if fitting fails.
    """
    t, y = _validate_data(t, y)
    if t is None:
        return None

    # Initial estimates
    K_init = np.max(y)
    y0_init = np.min(y)  # Baseline OD
    dy = np.gradient(y, t)
    t0_init = t[np.argmax(dy)]
    r_init = 0.01
    nu_init = 1.0  # Start with logistic-like shape

    p0 = [K_init, y0_init, r_init, t0_init, nu_init]
    bounds = (
        [y0_init * 0.5, 0, 0.0001, t.min(), 0.01],
        [np.inf, y0_init * 2, 10, t.max(), 100],
    )

    try:
        params, _ = curve_fit(richards_model, t, y, p0=p0, bounds=bounds, maxfev=20000)
        y_fit = richards_model(t, *params)
        # Calculate true specific growth rate from fitted curve
        mu_max = _calculate_specific_growth_rate(t, y_fit)
        return {
            "params": {
                "K": params[0],
                "y0": params[1],
                "r": params[2],
                "t0": params[3],
                "nu": params[4],
            },
            "y_fit": y_fit,
            "y": y,
            "t": t,
            "model_type": "richards",
            "mu_max": mu_max,
        }
    except Exception:
        return None


def fit_model(t, y, model_type="logistic"):
    """
    Fit a growth model to data.

    Parameters:
        t: Time array (hours)
        y: OD values
        model_type: One of "logistic", "gompertz", "richards"

    Returns:
        Fit result dict or None if fitting fails.
    """
    fit_funcs = {
        "logistic": fit_logistic,
        "gompertz": fit_gompertz,
        "richards": fit_richards,
    }
    fit_func = fit_funcs.get(model_type)
    if fit_func is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return fit_func(t, y)


# -----------------------------------------------------------------------------
# Growth Statistics Extraction
# -----------------------------------------------------------------------------


def _extract_stats_from_fit(fit_result, lag_frac=0.15, exp_frac=0.15):
    """
    Extract growth statistics from a model fit result.

    Parameters:
        fit_result: Dict from fit_* functions
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection

    Returns:
        Growth statistics dictionary.
    """
    if fit_result is None:
        return _bad_fit_stats()

    t = fit_result["t"]
    y = fit_result["y"]
    y_fit = fit_result["y_fit"]
    model_type = fit_result["model_type"]
    params = fit_result["params"]
    mu_max = fit_result["mu_max"]

    # Generate dense predictions for derivative calculation
    t_dense = np.linspace(t.min(), t.max(), 500)

    if model_type == "logistic":
        y_dense = logistic_model(
            t_dense, params["K"], params["y0"], params["r"], params["t0"]
        )
    elif model_type == "gompertz":
        y_dense = gompertz_model(
            t_dense, params["K"], params["y0"], params["mu_max_param"], params["lam"]
        )
    elif model_type == "richards":
        y_dense = richards_model(
            t_dense, params["K"], params["y0"], params["r"], params["t0"], params["nu"]
        )
    else:
        return _bad_fit_stats()

    # Calculate dN/dt for phase boundary detection
    dN_dt = np.gradient(y_dense, t_dense)

    # Maximum OD (carrying capacity from fit)
    max_od = float(params["K"])

    # Find time of maximum growth rate
    max_dN_idx = np.argmax(dN_dt)
    time_at_umax = float(t_dense[max_dN_idx])
    od_at_umax = float(y_dense[max_dN_idx])

    if mu_max <= 0:
        stats = _bad_fit_stats()
        stats["max_od"] = max_od
        stats["fit_method"] = f"model_fitting_{model_type}"
        return stats

    # Phase boundaries based on derivative thresholds
    max_dN = np.max(dN_dt)
    lag_threshold = lag_frac * max_dN
    exp_threshold = exp_frac * max_dN

    lag_idx = np.where(dN_dt >= lag_threshold)[0]
    exp_phase_start = (
        float(t_dense[lag_idx[0]]) if len(lag_idx) > 0 else float(t_dense[0])
    )

    exp_idx = np.where(
        (dN_dt <= exp_threshold) & (np.arange(len(t_dense)) > max_dN_idx)
    )[0]
    exp_phase_end = (
        float(t_dense[exp_idx[0]]) if len(exp_idx) > 0 else float(t_dense[-1])
    )

    # Doubling time based on mu_max (specific growth rate)
    doubling_time = np.log(2) / mu_max if mu_max > 0 else np.nan

    # RMSE in linear space
    rmse = _compute_rmse(y, y_fit)

    return {
        "max_od": max_od,
        "specific_growth_rate": float(mu_max),
        "doubling_time": float(doubling_time),
        "exp_phase_start": exp_phase_start,
        "exp_phase_end": max(exp_phase_end, exp_phase_start),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "t_window_start": float(t.min()),
        "t_window_end": float(t.max()),
        "fit_method": f"model_fitting_{model_type}",
        "model_rmse": rmse,
    }


def _bad_fit_stats():
    """Return default stats for failed fits."""
    return {
        "max_od": 0.0,
        "specific_growth_rate": 0.0,
        "doubling_time": np.nan,
        "exp_phase_start": np.nan,
        "exp_phase_end": np.nan,
        "time_at_umax": np.nan,
        "od_at_umax": np.nan,
        "t_window_start": np.nan,
        "t_window_end": np.nan,
        "fit_method": None,
        "model_rmse": np.nan,
    }


# -----------------------------------------------------------------------------
# Main API Functions
# -----------------------------------------------------------------------------


def fit_growth_model(t, y, model_type="logistic", lag_frac=0.15, exp_frac=0.15):
    """
    Fit a growth model and extract growth statistics.

    Parameters:
        t: Time array (hours)
        y: OD values (baseline-corrected)
        model_type: "logistic", "gompertz", or "richards"
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection

    Returns:
        Dict containing:
            - max_od: Maximum OD value (carrying capacity K)
            - specific_growth_rate: Maximum specific growth rate mu_max (h^-1)
            - doubling_time: Doubling time (hours)
            - exp_phase_start: Time when lag phase ends (hours)
            - exp_phase_end: Time when exponential phase ends (hours)
            - time_at_umax: Time at maximum growth rate (hours)
            - od_at_umax: OD at maximum growth rate
            - fit_method: Method used for fitting
            - model_rmse: Root mean square error of fit
    """
    fit_result = fit_model(t, y, model_type=model_type)
    return _extract_stats_from_fit(fit_result, lag_frac=lag_frac, exp_frac=exp_frac)


def sliding_window_fit(
    t, y, window_points=15, sg_window=11, sg_poly=1, lag_frac=0.15, exp_frac=0.15
):
    """
    Calculate growth statistics using the sliding window method.

    This method finds the maximum specific growth rate by fitting a line to
    log-transformed OD data in consecutive windows, selecting the window with
    the steepest slope. The slope of ln(OD) vs time gives the specific growth
    rate directly.

    Parameters:
        t: Time array (hours)
        y: OD values (baseline-corrected, must be positive)
        window_points: Number of points in each sliding window
        sg_window: Savitzky-Golay filter window size for smoothing
        sg_poly: Polynomial order for Savitzky-Golay filter
        lag_frac: Fraction of peak growth rate for lag phase detection
        exp_frac: Fraction of peak growth rate for exponential phase end detection

    Returns:
        Dict containing:
            - max_od: Maximum OD value
            - specific_growth_rate: Maximum specific growth rate (h^-1)
            - doubling_time: Doubling time (hours)
            - exp_phase_start: Time when lag phase ends (hours)
            - exp_phase_end: Time when exponential phase ends (hours)
            - time_at_umax: Time at maximum growth rate (hours)
            - od_at_umax: OD at maximum growth rate
            - t_window_start: Start time of best-fit window
            - t_window_end: End time of best-fit window
            - fit_method: Method used for fitting
            - model_rmse: RMSE of linear fit to log-transformed data in the window
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filter valid data (y must be positive for log transform)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    t, y_raw = t[mask], y[mask]

    if len(t) < max(10, window_points) or np.ptp(t) <= 0:
        return _bad_fit_stats()

    # Log-transform for growth rate calculation
    y_log = np.log(y_raw)

    # Smooth the log-transformed data for phase boundary detection
    y_log_smooth = _smooth(y_log, sg_window, sg_poly)

    # Maximum OD from raw data
    max_od = float(np.max(y_raw))

    # Find window with maximum slope on log-transformed data
    w = min(window_points, len(t))
    best_slope = -np.inf
    best_result = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, w)

    for i in range(len(t) - w + 1):
        t_win = t[i : i + w]
        y_log_win = y_log[i : i + w]

        if np.ptp(t_win) <= 0:
            continue

        slope, intercept = np.polyfit(t_win, y_log_win, 1)
        t_mid = float(np.mean(t_win))

        if slope > best_slope:
            best_slope = slope
            y_log_mid = slope * t_mid + intercept
            best_result = (
                t_mid,
                y_log_mid,
                float(t_win[0]),
                float(t_win[-1]),
                slope,
                intercept,
                i,
                i + w,
            )

    (
        time_at_umax,
        y_log_at_umax,
        t_window_start,
        t_window_end,
        slope,
        intercept,
        win_start,
        win_end,
    ) = best_result

    if not np.isfinite(best_slope) or best_slope <= 0:
        stats = _bad_fit_stats()
        stats["max_od"] = max_od
        return stats

    # Convert od_at_umax back to linear OD space
    od_at_umax = float(np.exp(y_log_at_umax))

    # Calculate RMSE for the linear fit on log-transformed data in the window
    t_win = t[win_start:win_end]
    y_log_win = y_log[win_start:win_end]
    y_log_pred = slope * t_win + intercept
    rmse = _compute_rmse(y_log_win, y_log_pred)

    # Calculate phase boundaries using derivative of smoothed log data
    lag_end, exp_end = _calculate_phase_ends(t, y_log_smooth, lag_frac, exp_frac)

    # Doubling time: t_d = ln(2) / mu
    doubling_time = np.log(2) / best_slope if best_slope > 0 else np.nan

    return {
        "max_od": max_od,
        "specific_growth_rate": float(best_slope),
        "doubling_time": float(doubling_time),
        "exp_phase_start": float(lag_end),
        "exp_phase_end": float(exp_end),
        "time_at_umax": time_at_umax,
        "od_at_umax": od_at_umax,
        "t_window_start": t_window_start,
        "t_window_end": t_window_end,
        "fit_method": "sliding_window",
        "model_rmse": rmse,
    }


def _calculate_phase_ends(t, y_smooth, lag_frac=0.15, exp_frac=0.15):
    """
    Calculate lag and exponential phase end times from smoothed data.

    Uses derivative analysis to identify phase transitions.
    """
    if len(t) < 5 or np.ptp(t) <= 0:
        return float(t[0]) if len(t) > 0 else np.nan, (
            float(t[-1]) if len(t) > 0 else np.nan
        )

    dy = np.gradient(y_smooth, t)
    dy = np.maximum(dy, 0)  # Only consider positive growth

    peak_idx = np.argmax(dy)
    peak_val = dy[peak_idx]

    if peak_val <= 0:
        return float(t[0]), float(t[-1])

    lag_threshold = lag_frac * peak_val
    exp_threshold = exp_frac * peak_val

    lag_idx = np.where(dy >= lag_threshold)[0]
    lag_end = float(t[lag_idx[0]]) if len(lag_idx) > 0 else float(t[0])

    exp_idx = np.where((dy <= exp_threshold) & (np.arange(len(t)) > peak_idx))[0]
    exp_end = float(t[exp_idx[0]]) if len(exp_idx) > 0 else float(t[-1])

    return lag_end, max(exp_end, lag_end)
