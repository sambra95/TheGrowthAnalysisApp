"""Plotting utilities for growth curves, stats, and window fits."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from functions.data_processing import ALL_WELLS, smooth


# --- helpers ------------------------------------------------------------------
def is_bad_fit(gs: dict) -> bool:
    """Return True when growth stats indicate a failed or missing fit."""
    mu = gs.get("Maximum U", gs.get("B", 0.0)) if gs else None
    return not gs or mu is None or (mu == 0.0)


def _finite_sorted_xy(time_s, y_s):
    """Return finite x/y arrays sorted by x."""
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


# --- blanks ----------------------------------------------------------------
def plot_baseline(baseline, name_by_well: dict | None = None):
    """Plot blank wells and mean baseline over time.

    Args:
        baseline: DataFrame with Time index and wells as columns (plus 'Mean' column)
        name_by_well: Optional dict mapping well IDs to sample names for color coding
    """
    fig = go.Figure()

    # Build color map based on sample names if available
    name_by_well = name_by_well or {}
    palette = px.colors.qualitative.Plotly

    # Get unique sample names from the wells in baseline (excluding 'Mean')
    well_cols = [c for c in baseline.columns if c != "Mean"]
    sample_names = list(dict.fromkeys([name_by_well.get(w, w) for w in well_cols]))
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(sample_names)}

    for col in baseline.columns:
        sample_name = name_by_well.get(col, col) if col != "Mean" else "Mean"
        color = color_map.get(sample_name, "black")

        fig.add_scatter(
            x=baseline.index,
            y=baseline[col],
            mode="markers" if col != "Mean" else "lines+markers",
            name=sample_name if col != "Mean" else "Mean",
            marker=dict(color=color) if col != "Mean" else None,
            line=dict(color=color) if col == "Mean" else None,
        )
    fig.update_yaxes(showgrid=False)

    return fig


# --- replicates ----------------------------------------------------------------
def _order_and_colors(d: pd.DataFrame, sample_order: list[str] | None):
    """Return ordered sample names and a deterministic color map."""
    ordered = list(dict.fromkeys([n for n in (sample_order or []) if n]))
    seen = list(pd.unique(d["Sample Name"]))
    names = ordered + [n for n in seen if n not in ordered]
    palette = px.colors.qualitative.Plotly
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    return names, color_map


def plot_replicates_scatter(
    curves_df: pd.DataFrame,
    sample_order: list[str] | None = None,
    t_start=0.0,
    t_end=72.0,
):
    """Scatter plot of replicate curves for selected samples and time window."""
    fig = go.Figure()
    fig.update_layout(
        xaxis_title="Time (hours)", yaxis_title="OD600 (baseline-corrected)", height=600
    )
    if curves_df is None or curves_df.empty:
        return fig

    d = curves_df[(curves_df["Time"] >= t_start) & (curves_df["Time"] <= t_end)].copy()
    names, color_map = _order_and_colors(d, sample_order)

    return px.scatter(
        d,
        x="Time",
        y="baseline_corrected",
        color="Sample Name",
        hover_data=["plate", "well", "key"],
        category_orders={"Sample Name": names},
        color_discrete_map=color_map,
    ).update_layout(
        height=600, xaxis_title="Time (hours)", yaxis_title="OD600 (baseline-corrected)"
    )


def plot_mean_growth(
    curves_df: pd.DataFrame, sample_order: list[str] | None, t_start=0.0, t_end=72.0
):
    """Plot mean curve with +/-1 SD shading for each sample."""
    fig = go.Figure()
    fig.update_layout(
        xaxis_title="Time (hours)", yaxis_title="OD600 (baseline-corrected)", height=600
    )
    if curves_df is None or curves_df.empty:
        return fig

    d = curves_df[(curves_df["Time"] >= t_start) & (curves_df["Time"] <= t_end)].copy()
    names, color_map = _order_and_colors(d, sample_order)

    agg = (
        d.groupby(["Sample Name", "Time"], as_index=False)["baseline_corrected"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
    )
    agg["upper"] = agg["mean"] + agg["sd"]
    agg["lower"] = agg["mean"] - agg["sd"]

    for nm in names:
        sub = agg[agg["Sample Name"] == nm].sort_values("Time")
        if sub.empty:
            continue
        c = color_map[nm]

        fig.add_trace(
            go.Scatter(
                x=pd.concat([sub["Time"], sub["Time"][::-1]]),
                y=pd.concat([sub["upper"], sub["lower"][::-1]]),
                fill="toself",
                fillcolor=c,
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
                line=dict(color=c),
                text=[nm] * len(sub),
                hovertemplate="Sample=%{text}<br>Time=%{x:.2f} h<br>Mean=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(legend_traceorder="normal")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_replicates_by_sample(plates: dict):
    """Create a grid of replicate scatter plots grouped by sample."""
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


# --- growth stats ----------------------------------------------------------------
def plot_growth_stats(
    long_df: pd.DataFrame,
    *,
    x_col: str = "Sample Name",
    legend_col: str | None = None,  # "Strain" / "Condition" / None
    x_order: list[str] | None = None,
    legend_order: list[str] | None = None,
):
    """Plot growth stats across samples with optional strain/condition splits."""

    if long_df is None or long_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Growth statistics", height=400)
        return fig

    metrics = [
        "Maximum U",
        "Maximum OD600",
        "Lag Time (hours)",
        "lag_phase_end",
        "exponential_phase_end",
        "t_mu",
        "y_mu",
        "b",
        "t_peak",
    ]

    df = long_df.copy()

    # ---- optionally derive Strain/Condition from sample_name (split on FIRST underscore) ----
    s = df["sample_name"].astype(str)
    has_split = s.str.contains("_", regex=False).any()
    if has_split:
        sc = s.str.split("_", n=1, expand=True)
        df["Strain"] = sc[0]
        df["Condition"] = sc[1].fillna("")

    # ---- map UI names to actual df columns ----
    col_map = {
        "Sample Name": "sample_name",
        "Strain": "Strain",
        "Condition": "Condition",
    }

    if x_col not in col_map:
        x_col = "Sample Name"
    x_df_col = col_map[x_col]

    legend_df_col = None
    if legend_col is not None:
        if legend_col not in col_map:
            legend_col = None
        else:
            legend_df_col = col_map[legend_col]
            # If user chose Strain/Condition but we didn't create them, drop back safely
            if legend_df_col not in df.columns:
                legend_col = None
                legend_df_col = None

    # If x axis requested Strain/Condition but absent, fall back to Sample Name
    if x_df_col not in df.columns:
        x_col = "Sample Name"
        x_df_col = "sample_name"

    # ---- x order categorical for stable ordering ----
    if not x_order:
        x_order = list(pd.unique(df[x_df_col].astype(str)))

    df["_x"] = (
        df[x_df_col]
        .astype(str)
        .astype(pd.CategoricalDtype(categories=list(x_order), ordered=True))
    )

    # ---- legend groups + stable color mapping ----
    if legend_col is None:
        groups: list[str] = []
        color_map: dict[str, str] = {}
    else:
        assert legend_df_col is not None
        if not legend_order:
            legend_order = list(pd.unique(df[legend_df_col].astype(str)))

        present = set(df[legend_df_col].astype(str))

        # enforce requested legend order
        groups = [str(g) for g in legend_order if str(g) in present]

        # include any stragglers at end
        for g in pd.unique(df[legend_df_col].astype(str)):
            gs = str(g)
            if gs not in groups:
                groups.append(gs)

        base_colors = px.colors.qualitative.Plotly  # change if you like
        color_map = {g: base_colors[i % len(base_colors)] for i, g in enumerate(groups)}

    # ---- aggregate ----
    group_cols = ["_x", "metric"]
    if legend_col is not None:
        group_cols.insert(1, legend_df_col)  # type: ignore[arg-type]

    agg = (
        df.groupby(group_cols, as_index=False)["value"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
        .sort_values(group_cols, kind="stable")
    )

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        subplot_titles=metrics,
    )

    for r, m in enumerate(metrics, 1):
        a = agg[agg["metric"] == m].copy()
        p = df[df["metric"] == m].copy()

        if legend_col is None:
            # Single series (no legend grouping)
            fig.add_trace(
                go.Bar(
                    x=a["_x"],
                    y=a["mean"],
                    error_y=dict(type="data", array=a["sd"], visible=True),
                    name="",
                    showlegend=False,
                    hovertemplate=f"{x_col}=%{{x}}<br>Mean=%{{y:.4f}}<extra></extra>",
                    marker=dict(line=dict(color="black", width=1.5)),
                ),
                row=r,
                col=1,
            )

            fig.add_trace(
                go.Box(
                    x=p["_x"],
                    y=p["value"].to_numpy(float),
                    name="",
                    showlegend=False,
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    fillcolor="rgba(0,0,0,0)",
                    line=dict(width=0),
                    marker=dict(size=6, opacity=0.8),
                    text=(
                        p["plate"].astype(str) + " " + p["well"].astype(str)
                    ).tolist(),
                    hovertemplate="Well=%{text}<br>Value=%{y:.4f}<extra></extra>",
                ),
                row=r,
                col=1,
            )
        else:
            # One legend entry per group, in user-defined order, with stable colors across all subplots
            assert legend_df_col is not None

            for g in groups:
                a_g = a[a[legend_df_col].astype(str) == g]
                p_g = p[p[legend_df_col].astype(str) == g]

                fig.add_trace(
                    go.Bar(
                        x=a_g["_x"],
                        y=a_g["mean"],
                        error_y=dict(type="data", array=a_g["sd"], visible=True),
                        name=g,
                        legendgroup=g,
                        offsetgroup=g,
                        showlegend=(r == 1),
                        hovertemplate=(
                            f"{legend_col}=%{{fullData.name}}<br>"
                            f"{x_col}=%{{x}}<br>"
                            "Mean=%{y:.4f}<extra></extra>"
                        ),
                        marker=dict(
                            color=color_map[g],
                            line=dict(color="black", width=1.5),
                        ),
                    ),
                    row=r,
                    col=1,
                )

                fig.add_trace(
                    go.Box(
                        x=p_g["_x"],
                        y=p_g["value"].to_numpy(float),
                        name=g,
                        legendgroup=g,
                        offsetgroup=g,
                        showlegend=False,
                        boxpoints="all",
                        jitter=0.35,
                        pointpos=0,
                        fillcolor="rgba(0,0,0,0)",
                        line=dict(width=0),
                        marker=dict(
                            color="black",
                            size=6,
                            opacity=0.8,
                        ),
                        text=(
                            p_g["plate"].astype(str) + " " + p_g["well"].astype(str)
                        ).tolist(),
                        hovertemplate="Well=%{text}<br>Value=%{y:.4f}<extra></extra>",
                    ),
                    row=r,
                    col=1,
                )

        fig.update_xaxes(
            showgrid=False,
            type="category",
            categoryorder="array",
            categoryarray=list(x_order),
            row=r,
            col=1,
            title_text=x_col if r == 3 else None,
        )
        fig.update_yaxes(showgrid=False, range=[0, None], row=r, col=1)

    fig.update_layout(
        title="Growth statistics",
        height=1000 * len(metrics),
        margin=dict(t=60),
        barmode="group",
        boxmode="group",
        legend_title_text=(legend_col if legend_col else ""),
        showlegend=bool(legend_col),
    )
    return fig


# --- window fits ----------------------------------------------------------------
def add_window_well(
    fig,
    *,
    d,  # dataframe for this well (or None/empty)
    well: str,
    gs: dict | None = None,
    row: int | None = None,
    col: int | None = None,
    line_hours: float = 2.0,
    marker_size: int = 5,
    marker_color: str = "red",
    line_color: str = "blue",
    shade_lag="rgba(180,180,180,0.18)",
    shade_exp="rgba(100,149,237,0.16)",
    shade_stat="rgba(144,238,144,0.16)",
    add_phase_shading: bool = True,
    add_window_line: bool = True,
):
    """
    Draw a single well (points + optional phase shading + optional window line)
    onto an existing Plotly figure.

    Works for both:
      - go.Figure() (row/col None)
      - make_subplots() figure (row/col provided)
    """
    gs = gs or {}

    # Helper: where to add traces
    trace_kwargs = {}
    if row is not None and col is not None:
        trace_kwargs = dict(row=row, col=col)

    # Empty: nothing to draw
    if d is None or d.empty:
        return

    t, y = _finite_sorted_xy(
        d["Time"].to_numpy(),
        d["baseline_corrected"].to_numpy(),
    )
    if t.size == 0:
        return

    tmin, tmax = float(t[0]), float(t[-1])

    # ---- Phase shading (needs correct xref/yref for each subplot) ----
    bad = is_bad_fit(gs)
    if add_phase_shading and (not bad):
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

        # IMPORTANT: xref/yref differ between single-figure and subplots
        if row is None:
            xref = "x"
            yref = "y domain"
        else:
            axis_index = (row - 1) * 12 + col  # for 8x12 plate only
            xref = "x" if axis_index == 1 else f"x{axis_index}"
            yref = "y domain" if axis_index == 1 else f"y{axis_index} domain"

        for x0, x1, colr in (
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
                fillcolor=colr,
                line_width=0,
                layer="below",
            )

    # ---- Scatter points ----
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="markers",
            marker=dict(size=marker_size, color=marker_color),
            hovertemplate=(
                f"Well={well}<br>Time=%{{x:.2f}} h<br>OD=%{{y:.4f}}<extra></extra>"
            ),
            showlegend=False,
        ),
        **trace_kwargs,
    )

    # ---- Window/gradient line ----
    if add_window_line:
        m = float(gs.get("Maximum U", gs.get("B", 0.0)) or 0.0)
        t0 = gs.get("t_mu")
        b0 = gs.get("b")
        if t0 is not None and b0 is not None and np.isfinite(m) and np.isfinite(t0) and np.isfinite(b0):
            t0 = float(t0)
            b0 = float(b0)
            x0, x1 = t0 - line_hours, t0 + line_hours
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[m * x0 + b0, m * x1 + b0],
                    mode="lines",
                    line=dict(width=2, color=line_color),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                **trace_kwargs,
            )


@st.cache_data(show_spinner=False)
def plot_window_single(
    processed_data: dict, well: str, plot_bgcolor="white", paper_bgcolor="white"
):
    """Plot a single well with lasso selection enabled."""
    d = (processed_data or {}).get(well)
    fig = go.Figure()

    add_window_well(
        fig,
        d=d,
        well=well,
        gs=None,  # or pass stats if you want shading/line here too
        row=None,
        col=None,
        marker_size=5,
    )

    # layout only here
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


def plot_window_plate(plate: dict, line_hours=2.0):
    """Plot a full 96-well plate overview with window-fit overlays."""
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

    # global ranges (same as you already do)
    ts, ys = [], []
    for d in proc.values():
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

    for i, well in enumerate(ALL_WELLS, 1):
        d = proc.get(well)
        if d is None or d.empty:
            continue

        r, c = divmod(i - 1, 12)
        r, c = r + 1, c + 1

        add_window_well(
            fig,
            d=d,
            well=well,
            gs=gs_all.get(well) or {},
            row=r,
            col=c,
            line_hours=line_hours,
            marker_size=2,  # plate: smaller dots
        )

    fig.update_layout(height=900, margin=dict(t=60), showlegend=False)
    fig.update_xaxes(showgrid=False, range=x_range, matches="x")
    fig.update_yaxes(showgrid=False, range=y_range, matches="y")
    return fig


def _vlines(
    fig, processed_data: dict, well: str, *xs, gs=None, line_hours: float = 4.0
):
    """Add phase shading, phase lines, and fit line annotations to a figure."""
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
        max_od = float(gs.get("Maximum OD600", 0.0) or 0.0)

        # colour code lag phase
        fig.add_vrect(
            x0=tmin,
            x1=lag_end,
            fillcolor="rgba(180,180,180,0.18)",
            line_width=0,
            layer="below",
        )

        # add line for lag end
        fig.add_vline(x=lag_end, line_dash="dot")

        # colour code exponential phase
        fig.add_vrect(
            x0=lag_end,
            x1=exp_end,
            fillcolor="rgba(100,149,237,0.16)",
            line_width=0,
            layer="below",
        )

        # add line for exp end
        fig.add_vline(x=exp_end, line_dash="dot")

        # colour code stationary phase
        fig.add_vrect(
            x0=exp_end,
            x1=tmax,
            fillcolor="rgba(144,238,144,0.16)",
            line_width=0,
            layer="below",
        )

        # add line for max OD600
        fig.add_hline(y=max_od, line_dash="dot")

        # fitted max gradient line in blue (constant geometric length)
        m = float(gs.get("Maximum U", 0.0) or 0.0)
        t0, b0 = gs.get("t_mu"), gs.get("b")
        if t0 is not None and b0 is not None and np.isfinite(m) and np.isfinite(t0) and np.isfinite(b0):
            t0, b0 = float(t0), float(b0)

            # line_hours now means "half-length" in Euclidean (data) units for x=hours, y=OD
            # segment half-length L, choose dx so sqrt(dx^2 + (m*dx)^2) = L  -> dx = L / sqrt(1+m^2)
            L = float(line_hours)
            dx = L / np.sqrt(1.0 + m * m)

            x0 = max(tmin, t0 - dx)
            x1 = min(tmax, t0 + dx)

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

    return fig


# --- derivative models ---------------------------------------------------------
def d1_model(t, A, r, t0):
    """Idealized first-derivative model for growth curves."""
    u = np.exp(-r * (t - t0))
    return A * (u / (1 + u) ** 2)


def d2_model(t, A, r, t0):
    """Idealized second-derivative model for growth curves."""
    u = np.exp(-r * (t - t0))
    return A * r * (u * (u - 1) / (1 + u) ** 3)


@st.cache_data
def _fit_idealised_derivatives(t, dy):
    """Fit the idealized derivative model to a gradient series."""
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
    plate: dict, well: str, sg_window=11, sg_poly=2, frac_peak=0.15, add_fit=True
):
    """Plot the first derivative of a well's smoothed curve."""
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t = d["Time"].to_numpy(float)
    y = d["baseline_corrected"].to_numpy(float)

    y_s = smooth(y, sg_window, sg_poly)
    dy = np.gradient(y_s, t)

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
    if not add_fit:
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


def plot_window_single_d2(
    plate: dict, well: str, sg_window=11, sg_poly=2, add_fit=True
):
    """Plot the second derivative of a well's smoothed curve."""
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t = d["Time"].to_numpy(float)
    y = d["baseline_corrected"].to_numpy(float)

    y_s = smooth(y, sg_window, sg_poly)
    dy = np.gradient(y_s, t)
    d2y = np.gradient(dy, t)

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

    if not add_fit:
        popt = _fit_idealised_derivatives(t, dy)
        d2_fit = None
        if popt is not None:
            A, r, t0 = popt
            d2_fit = d2_model(t, A, r, t0)

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
