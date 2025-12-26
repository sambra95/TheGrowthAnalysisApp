# plotting_functions.py — refactored for NEW plates structure

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import streamlit as st

from data_processing import ALL_WELLS, smooth


# --- helpers ------------------------------------------------------------------
def is_bad_fit(gs: dict) -> bool:
    return not gs or float(gs.get("Maximum U", gs.get("B", 0.0)) or 0.0) <= 0.0


def _finite_sorted_xy(time_s, y_s):
    t = np.asarray(time_s, float)
    y = np.asarray(y_s, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size:
        o = np.argsort(t)
        t, y = t[o], y[o]
    return t, y


def _iter_wells(plates: dict):
    """Yield (plate_id, plate, well, name, processed_df, growth_stats)."""
    for pid, p in plates.items():
        nm_by_well = p.get("name") or {}
        proc = p.get("processed_data") or {}
        gs_all = p.get("growth_stats") or {}
        for well, d in proc.items():
            yield pid, p, well, (nm_by_well.get(well) or ""), d, (
                gs_all.get(well) or {}
            )


# --- replicates ----------------------------------------------------------------
def plot_replicates_by_sample(plates: dict):
    items = [(pid, well, nm, d) for pid, _, well, nm, d, _ in _iter_wells(plates)]
    names = sorted(
        {(nm or "").strip() for *_, nm, __ in items} - {"", "False", "BLANK"}
    )

    cols = int(np.sqrt(max(1, len(names)))) + 1
    rows = (len(names) + cols - 1) // cols
    pos = {n: divmod(i, cols) for i, n in enumerate(names)}

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=names,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
        x_title="Time (hours)",
        y_title="OD600 (baseline-corrected)",
    )

    pal = px.colors.qualitative.Plotly
    keys = sorted({f"{pid}_{well}" for pid, well, *_ in items})
    cmap = {k: pal[i % len(pal)] for i, k in enumerate(keys)}

    tmins, tmaxs, ymins, ymaxs = [], [], [], []
    for pid, well, nm, d in items:
        nm = (nm or "").strip()
        if nm not in pos or d is None or d.empty:
            continue
        r, c = pos[nm]
        key = f"{pid}_{well}"

        fig.add_trace(
            go.Scatter(
                x=d["Time"],
                y=d["baseline_corrected"],
                mode="markers",
                marker=dict(size=3, color=cmap[key]),
                hovertemplate=f"Sample: {nm}<br>Well: {well}<br>Hour: %{{x:.2f}}<br>OD: %{{y:.4f}}<extra></extra><br>Plate: {pid}",
                showlegend=False,
            ),
            row=r + 1,
            col=c + 1,
        )
        tmins.append(float(d["Time"].min()))
        tmaxs.append(float(d["Time"].max()))
        ymins.append(float(d["baseline_corrected"].min()))
        ymaxs.append(float(d["baseline_corrected"].max()))

    fig.update_layout(height=750)
    if tmins:
        fig.update_xaxes(showgrid=False, range=[min(tmins), max(tmaxs)])
    else:
        fig.update_xaxes(showgrid=False)
    if ymins:
        fig.update_yaxes(showgrid=False, range=[min(ymins), max(ymaxs)])
    else:
        fig.update_yaxes(showgrid=False)
    return fig


# --- mean growth ----------------------------------------------------------------
def plot_mean_growth(plates, sel, t_start=0.0, t_end=72.0):
    items = list(_iter_wells(plates))

    if sel:
        ordered_names = list(dict.fromkeys([n for n in sel if n]))
    else:
        ordered_names, seen = [], set()
        for _, _, _, nm, _, _ in items:
            nm = (nm or "").strip()
            if nm and nm not in ("False", "BLANK") and nm not in seen:
                seen.add(nm)
                ordered_names.append(nm)

    sel = set(ordered_names)

    rows = []
    for pid, _, well, nm, d, _ in items:
        nm = (nm or "").strip()
        if nm in sel and d is not None and not d.empty:
            rows.append(
                pd.DataFrame(
                    {"name": nm, "Time": d["Time"], "y": d["baseline_corrected"]}
                )
            )

    d = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["name", "Time", "y"])
    )
    d = d[(d["Time"] >= t_start) & (d["Time"] <= t_end)]

    agg = (
        d.groupby(["name", "Time"], as_index=False)["y"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
    )
    agg["upper"] = agg["mean"] + agg["sd"]
    agg["lower"] = agg["mean"] - agg["sd"]

    fig = go.Figure()
    for nm in ordered_names:
        sub = agg[agg["name"] == nm].sort_values("Time")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.concat([sub["Time"], sub["Time"][::-1]]),
                y=pd.concat([sub["upper"], sub["lower"][::-1]]),
                fill="toself",
                line=dict(width=0),
                opacity=0.2,
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["Time"],
                y=sub["mean"],
                mode="lines",
                name=nm,
                text=[nm] * len(sub),
                hovertemplate="Sample=%{text}<br>Time=%{x:.2f} h<br>Mean=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Time (hours)",
        yaxis_title="OD600 (baseline-corrected)",
        height=600,
        legend_traceorder="normal",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# --- growth stats ----------------------------------------------------------------
def plot_growth_stats(long_df: pd.DataFrame, sample_order: list[str]):
    """
    long_df columns expected: plate, well, sample_name, metric, value
    sample_order: list of sample names in desired x order
    """
    if long_df is None or long_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Growth statistics", height=400)
        return fig

    metrics = ["Maximum OD600", "Maximum U", "Lag Time (hours)"]
    long_df = long_df.copy()

    # enforce category order
    cat = pd.CategoricalDtype(
        categories=[s for s in sample_order if s in long_df["sample_name"].unique()],
        ordered=True,
    )
    long_df["sample_name"] = long_df["sample_name"].astype(cat)

    agg = (
        long_df.groupby(["sample_name", "metric"], as_index=False)["value"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
        .sort_values(["sample_name", "metric"], kind="stable")
    )

    fig = make_subplots(rows=3, cols=1, subplot_titles=metrics, vertical_spacing=0.08)

    for r, m in enumerate(metrics, 1):
        a = agg[agg["metric"] == m].sort_values("sample_name", kind="stable")
        p = long_df[long_df["metric"] == m].sort_values("sample_name", kind="stable")

        fig.add_trace(
            go.Bar(
                x=a["sample_name"],
                y=a["mean"],
                error_y=dict(type="data", array=a["sd"], visible=True),
                hovertemplate="Sample=%{x}<br>Mean=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=1,
        )
        fig.add_trace(
            go.Box(
                x=p["sample_name"],
                y=p["value"].to_numpy(float),
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                fillcolor="rgba(0,0,0,0)",
                line=dict(width=0),
                marker=dict(size=6, opacity=0.8),
                text=(p["plate"].astype(str) + " " + p["well"].astype(str)).tolist(),
                hovertemplate="Well=%{text}<br>Value=%{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=1,
        )

        fig.update_xaxes(
            showgrid=False,
            categoryorder="array",
            categoryarray=sample_order,
            row=r,
            col=1,
        )
        fig.update_yaxes(showgrid=False, range=[0, None], row=r, col=1)

    fig.update_layout(
        title="Growth statistics", height=1400, margin=dict(t=60), showlegend=False
    )
    return fig


# --- window fits ----------------------------------------------------------------
def plot_window_plate(plate: dict, line_hours=2.0):
    proc = plate.get("processed_data") or {}
    gs_all = plate.get("growth_stats") or {}

    fig = make_subplots(
        rows=8,
        cols=12,
        horizontal_spacing=0.004,
        vertical_spacing=0.03,
        subplot_titles=ALL_WELLS,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    # global ranges
    ts, ys = [], []
    for w, d in proc.items():
        if d is None or d.empty:
            continue
        ts.append(d["Time"])
        ys.append(d["baseline_corrected"])
    if not ts:
        fig.update_layout(height=900, margin=dict(t=60), showlegend=False)
        return fig

    x_min, x_max = float(min(t.min() for t in ts)), float(max(t.max() for t in ts))
    y_min, y_max = float(min(y.min() for y in ys)), float(max(y.max() for y in ys))
    xr, yr = x_max - x_min, y_max - y_min
    x_range = [x_min - 0.02 * xr, x_max + 0.02 * xr]
    y_range = [y_min - 0.05 * yr, y_max + 0.05 * yr]

    shade_lag, shade_exp, shade_stat = (
        "rgba(180,180,180,0.18)",
        "rgba(100,149,237,0.16)",
        "rgba(144,238,144,0.16)",
    )

    for i, w in enumerate(ALL_WELLS, 1):
        d = proc.get(w)
        if d is None or d.empty:
            continue

        r, c = divmod(i - 1, 12)
        r, c = r + 1, c + 1

        t = d["Time"].to_numpy(float)
        y = d["baseline_corrected"].to_numpy(float)
        tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))

        gs = gs_all.get(w) or {}
        bad = is_bad_fit(gs)

        if not bad:
            lag_end = float(
                np.clip(gs.get("lag_phase_end", gs.get("lag_end", tmin)), tmin, tmax)
            )
            exp_end = float(
                np.clip(
                    gs.get("exponential_phase_end", gs.get("exp_end", tmax)), tmin, tmax
                )
            )
            if exp_end < lag_end:
                exp_end = lag_end

            xref = "x" if i == 1 else f"x{i}"
            yref = "y domain" if i == 1 else f"y{i} domain"
            for x0, x1, col in (
                (tmin, lag_end, shade_lag),
                (lag_end, exp_end, shade_exp),
                (exp_end, tmax, shade_stat),
            ):
                fig.add_shape(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    xref=xref,
                    yref=yref,
                    fillcolor=col,
                    line_width=0,
                    layer="below",
                )

        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="markers",
                marker=dict(size=2, color="red"),  # <-- add color
                hovertemplate=f"Well={w}<br>Time=%{{x:.2f}} h<br>OD=%{{y:.4f}}<extra></extra>",
                showlegend=False,
            ),
            r,
            c,
        )

        # red window line from stats (t_mu, b, Maximum U)
        m = float(gs.get("Maximum U", gs.get("B", 0.0)) or 0.0)
        t0 = gs.get("t_mu")
        b0 = gs.get("b")
        if np.isfinite(m) and m > 0 and np.isfinite(t0) and np.isfinite(b0):
            t0 = float(t0)
            b0 = float(b0)
            x0, x1 = t0 - line_hours, t0 + line_hours
            # fitted max gradient line: force blue
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[m * x0 + b0, m * x1 + b0],
                    mode="lines",
                    line=dict(width=2, color="blue"),  # <-- add color
                    hoverinfo="skip",
                    showlegend=False,
                ),
                r,
                c,
            )

    fig.update_layout(height=900, margin=dict(t=60), showlegend=False)
    fig.update_xaxes(showgrid=False, range=x_range, matches="x")
    fig.update_yaxes(showgrid=False, range=y_range, matches="y")
    return fig


@st.cache_data(show_spinner=False)
def plot_window_single(
    processed_data: dict,
    well: str,
    plot_bgcolor: str = "white",
    paper_bgcolor: str = "white",
):
    d = (processed_data or {}).get(well)

    fig = go.Figure()

    if d is None or d.empty:
        fig.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            uirevision="keep",
            dragmode="lasso",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_xaxes(type="linear", showgrid=False, title="Time (hours)")
        fig.update_yaxes(showgrid=False, title="OD600 (baseline-corrected)")
        return fig

    t, y = _finite_sorted_xy(d["Time"].to_numpy(), d["baseline_corrected"].to_numpy())
    if t.size == 0:
        fig.update_layout(
            height=600,
            showlegend=False,
            plot_bgcolor=plot_bgcolor,
            paper_bgcolor=paper_bgcolor,
            uirevision="keep",
            dragmode="lasso",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_xaxes(type="linear", showgrid=False, title="Time (hours)")
        fig.update_yaxes(showgrid=False, title="OD600 (baseline-corrected)")
        return fig

    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="markers",
            marker=dict(size=5, color="red"),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} h<br>OD=%{{y:.4f}}<extra></extra>",
            showlegend=False,
        )
    )

    tmin, tmax = float(t[0]), float(t[-1])
    xr = (tmax - tmin) if np.isfinite(tmax - tmin) and (tmax - tmin) > 0 else 1.0
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        uirevision="keep",
        dragmode="lasso",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_xaxes(
        type="linear",
        range=[tmin - 0.02 * xr, tmax + 0.02 * xr],
        showgrid=False,
        title="Time (hours)",
    )
    fig.update_yaxes(showgrid=False, title="OD600 (baseline-corrected)")
    return fig


def _vlines(
    fig, processed_data: dict, well: str, *xs, gs=None, line_hours: float = 4.0
):
    # always start clean (important when reusing/copying figures)
    fig.update_layout(shapes=[])

    # compute range from the real data (NOT from fig.data which may be typed-array dicts)
    d = (processed_data or {}).get(well)
    if d is None or d.empty:
        return fig

    t, y = _finite_sorted_xy(d["Time"].to_numpy(), d["baseline_corrected"].to_numpy())
    if t.size == 0:
        return fig

    tmin, tmax = float(t[0]), float(t[-1])

    # --- shading + fit line from growth stats ---
    gs = gs or {}
    if gs and not is_bad_fit(gs):
        lag_end = float(np.clip(gs.get("lag_phase_end", tmin), tmin, tmax))
        exp_end = float(np.clip(gs.get("exponential_phase_end", tmax), tmin, tmax))
        exp_end = max(exp_end, lag_end)

        fig.add_vrect(
            x0=tmin,
            x1=lag_end,
            fillcolor="rgba(180,180,180,0.18)",
            line_width=0,
            layer="below",
        )
        fig.add_vrect(
            x0=lag_end,
            x1=exp_end,
            fillcolor="rgba(100,149,237,0.16)",
            line_width=0,
            layer="below",
        )
        fig.add_vrect(
            x0=exp_end,
            x1=tmax,
            fillcolor="rgba(144,238,144,0.16)",
            line_width=0,
            layer="below",
        )

        m = float(gs.get("Maximum U", 0.0) or 0.0)
        t0, b0 = gs.get("t_mu"), gs.get("b")
        if np.isfinite(m) and np.isfinite(t0) and np.isfinite(b0):
            t0, b0 = float(t0), float(b0)
            x0 = max(tmin, t0 - line_hours)
            x1 = min(tmax, t0 + line_hours)
            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=x0,
                y0=m * x0 + b0,
                x1=x1,
                y1=m * x1 + b0,
                line=dict(width=3, color="rgba(30, 144, 255, 0.7)"),
            )

    # --- phase boundary vertical lines ---
    for x in xs:
        if np.isfinite(x):
            try:
                fig.add_vline(x=float(x), line_dash="dot")
            except Exception:
                pass

    return fig


# --- derivative models ---------------------------------------------------------
def d1_model(t, A, r, t0):
    u = np.exp(-r * (t - t0))
    return A * (u / (1 + u) ** 2)


def d2_model(t, A, r, t0):
    u = np.exp(-r * (t - t0))
    return A * r * (u * (u - 1) / (1 + u) ** 3)


@st.cache_data
def _fit_idealised_derivatives(t, dy):
    mask = np.isfinite(t) & np.isfinite(dy)
    t_fit, dy_fit = t[mask], dy[mask]
    if t_fit.size < 10:
        return None

    t0_guess = float(t_fit[np.argmax(dy_fit)])
    dy_max = float(np.max(dy_fit))
    if not np.isfinite(dy_max) or dy_max <= 0:
        return None

    p0 = [4.0 * dy_max, 0.05, t0_guess]
    bounds = ([0.0, 1e-6, float(np.min(t_fit))], [np.inf, 10.0, float(np.max(t_fit))])

    try:
        popt, _ = curve_fit(d1_model, t_fit, dy_fit, p0=p0, bounds=bounds, maxfev=20000)
        return popt
    except Exception:
        return None


@st.fragment
def plot_window_single_d1(
    plate: dict, well: str, sg_window=11, sg_poly=2, frac_peak=0.15
):
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t = d["Time"].to_numpy(float)
    y = d["baseline_corrected"].to_numpy(float)

    y_s = smooth(y, sg_window, sg_poly)
    dy = np.gradient(y_s, t)

    popt = _fit_idealised_derivatives(t, dy)
    dy_fit = t_lag = t_exp = thr = None

    if popt is not None:
        A, r, t0 = popt
        dy_fit = d1_model(t, A, r, t0)
        if np.isfinite(dy_fit).any():
            peak_i = int(np.nanargmax(dy_fit))
            thr = frac_peak * float(dy_fit[peak_i])

            idx_up = np.where(dy_fit >= thr)[0]
            if idx_up.size:
                t_lag = float(t[idx_up[0]])

            idx_dn = np.where((dy_fit <= thr) & (np.arange(len(t)) > peak_i))[0]
            if idx_dn.size:
                t_exp = float(t[idx_dn[0]])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=dy,
            mode="lines",
            line=dict(width=2),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} h<br>dy/dt=%{{y:.6f}}<extra></extra>",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if dy_fit is not None and np.isfinite(dy_fit).any():
        fig.add_trace(
            go.Scatter(
                x=t,
                y=dy_fit,
                mode="lines",
                line=dict(width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"First derivative (smoothed) – {well}",
        height=320,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, title="Time (hours)")
    fig.update_yaxes(showgrid=False, title="d(OD)/dt")
    return fig


def plot_window_single_d2(plate: dict, well: str, sg_window=11, sg_poly=2):
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t = d["Time"].to_numpy(float)
    y = d["baseline_corrected"].to_numpy(float)

    y_s = smooth(y, sg_window, sg_poly)
    dy = np.gradient(y_s, t)
    d2y = np.gradient(dy, t)

    popt = _fit_idealised_derivatives(t, dy)
    d2_fit = None
    if popt is not None:
        A, r, t0 = popt
        d2_fit = d2_model(t, A, r, t0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=d2y,
            mode="lines",
            line=dict(width=2),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} h<br>d²y/dt²=%{{y:.6f}}<extra></extra>",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if d2_fit is not None and np.isfinite(d2_fit).any():
        fig.add_trace(
            go.Scatter(
                x=t,
                y=d2_fit,
                mode="lines",
                line=dict(width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"Second derivative (smoothed) – {well}",
        height=320,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, title="Time (hours)")
    fig.update_yaxes(showgrid=False, title="d²(OD)/dt²")
    return fig
