# data_processing.py
import io

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import streamlit as st

ROWS = "ABCDEFGH"
COLS = range(1, 13)
ALL_WELLS = [f"{r}{c}" for r in ROWS for c in COLS]


# ---------- small utilities ----------
def _as_float(x):
    return np.asarray(x, dtype=float)


@st.cache_data
def smooth(y, window=21, poly=1, passes=2):
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
    u = np.exp(-r * (t - t0))
    return A * (u / (1 + u) ** 2)


@st.cache_data
def fit_d1(t, dy):
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


def phase_ends(t, y_s, frac_peak=0.10):
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


# ---------- growth stats ----------
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
}


def window_fit(t, y, w=15, *, sg_window=11, sg_poly=1, lag_frac=0.10):
    t, y = _as_float(t), _as_float(y)
    w = int(w)

    if t.size < max(5, w) or np.ptp(t) <= 0:
        return BAD_FIT.copy()

    y_s = smooth(y, sg_window, sg_poly)
    m = np.isfinite(t) & np.isfinite(y_s)
    t, y_s = t[m], y_s[m]
    if t.size < max(5, w) or np.ptp(t) <= 0:
        return BAD_FIT.copy()

    dy = np.diff(y_s)
    mad = np.median(np.abs(dy - np.median(dy)))
    noise = 1.4826 * mad
    if noise == 0 or (y_s.max() - y_s.min()) <= 2 * noise:
        return BAD_FIT.copy()

    peak_i = int(np.nanargmax(y_s))
    t_peak, A = float(t[peak_i]), float(y_s[peak_i])

    w = min(w, t.size)
    best_m, best = -np.inf, (np.nan, np.nan, np.nan)  # (t_mu, y_mu, b)
    for i in range(t.size - w + 1):
        tw, yw = t[i : i + w], y_s[i : i + w]
        if np.ptp(tw) <= 0:
            continue
        m_i, b_i = np.polyfit(tw, yw, 1)
        t_mu = float(tw.mean())
        if t_mu <= t_peak and m_i > best_m:
            best_m = float(m_i)
            best = (t_mu, float(best_m * t_mu + b_i), float(b_i))

    t_mu, y_mu, b = best
    if not np.isfinite(best_m) or best_m <= 0:
        out = BAD_FIT.copy()
        out.update({"Maximum OD600": A, "t_peak": t_peak})
        return out

    lag_end, exp_end = phase_ends(t, y_s, frac_peak=lag_frac)

    out = BAD_FIT.copy()
    out.update(
        {
            "Maximum OD600": A,
            "Maximum U": float(best_m),
            "Lag Time (hours)": float(lag_end - t[0]),
            "lag_phase_end": float(lag_end),
            "exponential_phase_end": float(exp_end),
            "t_mu": float(t_mu),
            "y_mu": float(y_mu),
            "b": float(b),
            "t_peak": float(t_peak),
        }
    )
    return out


# ---------- I/O + shaping ----------
def _read_excel_bytes(b, **kw):
    return pd.read_excel(io.BytesIO(b), **kw)


def _plate_name_map(plate_bytes):
    plate = _read_excel_bytes(plate_bytes).fillna("False").set_index("rows")
    return plate, {f"{r}{c}": plate.loc[r, c] for r in ROWS for c in COLS}


def _read_table(data_bytes: bytes, read_interval_min: int) -> pd.DataFrame:
    """
    New format: rows=timepoints, cols=wells. Optional 'Time' column.
    Produces Time in hours.
    """
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
    return {"name": {}, "raw_data": {}, "processed_data": {}, "growth_stats": {}}


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
}


def load_plate(
    plates: dict, plate_id: str, *, data_bytes: bytes, plate_bytes: bytes, params: dict
):
    rec = plates.setdefault(plate_id, {})
    rec["uploads"] = {"data_bytes": data_bytes, "plate_bytes": plate_bytes}
    rec["params"] = params
    return rec


def analyse_plate(record: dict):
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

    plate = _empty_plate()
    plate["baseline"] = baseline
    plate["plate_map"] = plate_map

    for well, g in long.groupby("well", sort=False):
        processed = g[["Time", "baseline_corrected"]].reset_index(drop=True)

        try:
            fit = window_fit(
                processed["Time"].to_numpy(float),
                processed["baseline_corrected"].to_numpy(float),
                int(p["window_points"]),
                sg_window=int(p.get("sg_window", 11)),
                sg_poly=int(p.get("sg_poly", 2)),
                lag_frac=float(p.get("lag_frac", 0.20)),
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
    plates, window_points=15, sg_window=11, sg_poly=2, lag_frac=0.20
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
            fit = window_fit(
                d["Time"].to_numpy(float),
                d["baseline_corrected"].to_numpy(float),
                int(window_points),
                sg_window=int(sg_window),
                sg_poly=int(sg_poly),
                lag_frac=float(lag_frac),
            )

            wdict.update(
                {
                    "Maximum OD600": float(fit["A"]),
                    "Maximum U": float(fit["B"]),
                    "Lag Time (hours)": float(fit["C"]),
                    "lag_phase_end": float(fit["lag_end"]),
                    "exponential_phase_end": float(fit["exp_end"]),
                    "t_mu": float(fit["t_mu"]) if np.isfinite(fit["t_mu"]) else np.nan,
                    "y_mu": float(fit["y_mu"]) if np.isfinite(fit["y_mu"]) else np.nan,
                    "b": float(fit["b"]) if np.isfinite(fit["b"]) else np.nan,
                    "t_peak": (
                        float(fit["t_peak"]) if np.isfinite(fit["t_peak"]) else np.nan
                    ),
                }
            )

            plate_stats[well] = {
                "Sample Name": plate["name"].get(well, ""),
                "Maximum OD600": wdict["Maximum OD600"],
                "Maximum U": wdict["Maximum U"],
                "Lag Time (hours)": wdict["Lag Time (hours)"],
            }
        stats[plate_id] = plate_stats
    return stats
