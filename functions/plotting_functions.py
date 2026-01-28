"""Plotting utilities for growth curves, stats, and window fits."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit

from functions.data_processing import (
    ALL_WELLS,
    compute_first_derivative,
    compute_second_derivative,
    smooth,
)
from growthcurves.models import (
    gompertz_model,
    logistic_model,
    richards_model,
)
from growthcurves.fitting_functions import (
    fit_model,
    is_no_growth,
)


# --- time unit helpers --------------------------------------------------------
def get_time_label(time_unit: str = "hours") -> str:
    """Get the x-axis label for time based on the unit."""
    return f"Time ({time_unit})"


def convert_hours_to_unit(hours: float | np.ndarray, time_unit: str = "hours"):
    """Convert time from hours to the specified display unit.

    Args:
        hours: Time value(s) in hours
        time_unit: Target unit ("seconds", "minutes", or "hours")

    Returns:
        Time value(s) in the target unit
    """
    if time_unit == "seconds":
        return hours * 3600.0
    elif time_unit == "minutes":
        return hours * 60.0
    else:  # hours
        return hours


# --- helpers ------------------------------------------------------------------
# Alias for backward compatibility - use is_no_growth from python_package
def is_bad_fit(gs: dict) -> bool:
    """Return True when growth stats indicate a failed or missing fit."""
    return is_no_growth(gs)


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


def _model_type_from_fit_method(fit_method: str | None) -> str | None:
    """Extract model type from fit_method strings."""
    if not fit_method:
        return None
    if "(" in fit_method and ")" in fit_method:
        return fit_method.split("(", 1)[1].split(")", 1)[0].strip()
    if "model_fitting_" in fit_method:
        return fit_method.split("model_fitting_", 1)[1].strip()
    return None


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
def plot_baseline(baseline, name_by_well: dict | None = None, time_unit: str = "hours"):
    """Plot blank wells and mean baseline over time.

    Args:
        baseline: DataFrame with Time index and wells as columns (plus 'Mean' column)
        name_by_well: Optional dict mapping well IDs to sample names for color coding
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    fig = go.Figure()

    # Build color map based on sample names if available
    name_by_well = name_by_well or {}
    palette = px.colors.qualitative.Plotly

    # Get unique sample names from the wells in baseline (excluding 'Mean')
    well_cols = [c for c in baseline.columns if c != "Mean"]
    sample_names = list(dict.fromkeys([name_by_well.get(w, w) for w in well_cols]))
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(sample_names)}

    # Convert time index to display unit
    time_display = convert_hours_to_unit(baseline.index.to_numpy(), time_unit)

    for col in baseline.columns:
        sample_name = name_by_well.get(col, col) if col != "Mean" else "Mean"
        color = color_map.get(sample_name, "black")

        fig.add_scatter(
            x=time_display,
            y=baseline[col],
            mode="markers" if col != "Mean" else "lines+markers",
            name=sample_name if col != "Mean" else "Mean",
            marker=dict(color=color) if col != "Mean" else None,
            line=dict(color=color) if col == "Mean" else None,
        )
    fig.update_xaxes(title=get_time_label(time_unit))
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
    time_unit: str = "hours",
):
    """Scatter plot of replicate curves for selected samples and time window.

    Args:
        curves_df: DataFrame with Time, baseline_corrected, Sample Name columns
        sample_order: Optional list of sample names for ordering
        t_start: Start time for filtering (in hours)
        t_end: End time for filtering (in hours)
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    time_label = get_time_label(time_unit)
    fig = go.Figure()
    fig.update_layout(
        xaxis_title=time_label, yaxis_title="OD600 (baseline-corrected)", height=600
    )
    if curves_df is None or curves_df.empty:
        return fig

    d = curves_df[(curves_df["Time"] >= t_start) & (curves_df["Time"] <= t_end)].copy()
    # Convert time to display unit
    d["Time_display"] = convert_hours_to_unit(d["Time"].to_numpy(), time_unit)
    names, color_map = _order_and_colors(d, sample_order)

    return px.scatter(
        d,
        x="Time_display",
        y="baseline_corrected",
        color="Sample Name",
        hover_data=["plate", "well", "key"],
        category_orders={"Sample Name": names},
        color_discrete_map=color_map,
    ).update_layout(
        height=600, xaxis_title=time_label, yaxis_title="OD600 (baseline-corrected)"
    )


def plot_mean_growth(
    curves_df: pd.DataFrame,
    sample_order: list[str] | None,
    t_start=0.0,
    t_end=72.0,
    time_unit: str = "hours",
):
    """Plot mean curve with +/-1 SD shading for each sample.

    Args:
        curves_df: DataFrame with Time, baseline_corrected, Sample Name columns
        sample_order: Optional list of sample names for ordering
        t_start: Start time for filtering (in hours)
        t_end: End time for filtering (in hours)
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    time_label = get_time_label(time_unit)
    fig = go.Figure()
    fig.update_layout(
        xaxis_title=time_label, yaxis_title="OD600 (baseline-corrected)", height=600
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
    # Convert time to display unit
    agg["Time_display"] = convert_hours_to_unit(agg["Time"].to_numpy(), time_unit)

    for nm in names:
        sub = agg[agg["Sample Name"] == nm].sort_values("Time")
        if sub.empty:
            continue
        c = color_map[nm]

        fig.add_trace(
            go.Scatter(
                x=pd.concat([sub["Time_display"], sub["Time_display"][::-1]]),
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
                x=sub["Time_display"],
                y=sub["mean"],
                mode="lines",
                name=nm,
                line=dict(color=c),
                text=[nm] * len(sub),
                hovertemplate=f"Sample=%{{text}}<br>Time=%{{x:.2f}} {time_unit}<br>Mean=%{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(legend_traceorder="normal")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_replicates_by_sample(plates: dict, time_unit: str = "hours"):
    """Create a grid of replicate scatter plots grouped by sample.

    Args:
        plates: Dictionary of plate data
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    items = [(pid, well, nm, d) for pid, _, well, nm, d, _ in _iter_wells(plates)]
    names = sorted(
        {(nm or "").strip() for *_, nm, __ in items} - {"", "False", "BLANK"}
    )

    cols = int(np.sqrt(max(1, len(names)))) + 1
    rows = (len(names) + cols - 1) // cols
    pos = {n: divmod(i, cols) for i, n in enumerate(names)}

    time_label = get_time_label(time_unit)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=names,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.07,
        x_title=time_label,
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

        # Convert time to display unit
        time_display = convert_hours_to_unit(d["Time"].to_numpy(), time_unit)

        fig.add_trace(
            go.Scatter(
                x=time_display,
                y=d["baseline_corrected"],
                mode="markers",
                marker=dict(size=3, color=cmap[key]),
                hovertemplate=f"Sample: {nm}<br>Well: {well}<br>Time: %{{x:.2f}} {time_unit}<br>OD: %{{y:.4f}}<extra></extra><br>Plate: {pid}",
                showlegend=False,
            ),
            row=r + 1,
            col=c + 1,
        )
        tmins.append(float(time_display.min()))
        tmaxs.append(float(time_display.max()))
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
# Mapping of metric names to their units for y-axis labels
# Keys from python_package.py
METRIC_UNITS = {
    "specific_growth_rate": "h⁻¹",
    "doubling_time": "hours",
    "max_od": "OD600",
    "exp_phase_start": "hours",
    "exp_phase_end": "hours",
    "time_at_umax": "hours",
    "od_at_umax": "OD600",
}

# Mapping of metric names to display titles
METRIC_TITLES = {
    "specific_growth_rate": "Maximum specific growth rate",
    "doubling_time": "Doubling time",
    "max_od": "Maximum OD",
    "exp_phase_start": "Lag phase end",
    "exp_phase_end": "Exponential phase end",
    "time_at_umax": "Time at max growth rate",
    "od_at_umax": "OD at max growth rate",
}

# Mapping of metric names to y-axis labels (using Greek letters where appropriate)
METRIC_Y_LABELS = {
    "specific_growth_rate": "μ",
    "doubling_time": "tᵈ",
}


def plot_single_growth_stat(
    long_df: pd.DataFrame,
    *,
    x_col: str = "Sample Name",
    legend_col: str | None = None,
    x_order: list[str] | None = None,
    legend_order: list[str] | None = None,
):
    """Plot a single growth metric across samples with optional strain/condition splits."""

    if long_df is None or long_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Growth statistics", height=400)
        return fig

    df = long_df.copy()

    # Get the metric name from the data
    metric = (
        df["metric"].iloc[0] if "metric" in df.columns and not df.empty else "Metric"
    )

    # Get display title (defaults to metric name if not specified)
    display_title = METRIC_TITLES.get(metric, metric)

    # Get the y-axis label (use Greek letter if available, otherwise metric name)
    y_label = METRIC_Y_LABELS.get(metric, metric)
    unit = METRIC_UNITS.get(metric, "")
    y_axis_label = f"{y_label} ({unit})" if unit else y_label

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

        base_colors = px.colors.qualitative.Plotly
        color_map = {g: base_colors[i % len(base_colors)] for i, g in enumerate(groups)}

    # ---- aggregate ----
    group_cols = ["_x"]
    if legend_col is not None:
        group_cols.insert(0, legend_df_col)  # type: ignore[arg-type]

    agg = (
        df.groupby(group_cols, as_index=False)["value"]
        .agg(mean="mean", sd="std")
        .fillna({"sd": 0.0})
        .sort_values(group_cols, kind="stable")
    )

    # Create a single plot (not subplots)
    fig = go.Figure()

    if legend_col is None:
        # Single series (no legend grouping)
        fig.add_trace(
            go.Bar(
                x=agg["_x"],
                y=agg["mean"],
                error_y=dict(type="data", array=agg["sd"], visible=True),
                name="",
                showlegend=False,
                hovertemplate=f"{x_col}=%{{x}}<br>Mean=%{{y:.4f}}<extra></extra>",
                marker=dict(line=dict(color="black", width=1.5)),
            )
        )

        fig.add_trace(
            go.Box(
                x=df["_x"],
                y=df["value"].to_numpy(float),
                name="",
                showlegend=False,
                boxpoints="all",
                jitter=0.35,
                pointpos=0,
                fillcolor="rgba(0,0,0,0)",
                line=dict(width=0),
                marker=dict(size=6, opacity=0.8),
                text=(df["plate"].astype(str) + " " + df["well"].astype(str)).tolist(),
                hovertemplate="Well=%{text}<br>Value=%{y:.4f}<extra></extra>",
            )
        )
    else:
        # One legend entry per group, in user-defined order, with stable colors
        assert legend_df_col is not None

        for g in groups:
            a_g = agg[agg[legend_df_col].astype(str) == g]
            p_g = df[df[legend_df_col].astype(str) == g]

            fig.add_trace(
                go.Bar(
                    x=a_g["_x"],
                    y=a_g["mean"],
                    error_y=dict(type="data", array=a_g["sd"], visible=True),
                    name=g,
                    legendgroup=g,
                    offsetgroup=g,
                    showlegend=True,
                    hovertemplate=(
                        f"{legend_col}=%{{fullData.name}}<br>"
                        f"{x_col}=%{{x}}<br>"
                        "Mean=%{y:.4f}<extra></extra>"
                    ),
                    marker=dict(
                        color=color_map[g],
                        line=dict(color="black", width=1.5),
                    ),
                )
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
                )
            )

    fig.update_xaxes(
        showgrid=False,
        type="category",
        categoryorder="array",
        categoryarray=list(x_order),
        title_text=x_col,
    )
    fig.update_yaxes(showgrid=False, range=[0, None], title_text=y_axis_label)

    fig.update_layout(
        title=display_title,
        height=500,
        margin=dict(t=60, b=60),
        barmode="group",
        boxmode="group",
        legend_title_text=(legend_col if legend_col else ""),
        showlegend=bool(legend_col),
    )
    return fig


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

    # Keys from python_package.py
    metrics = [
        "specific_growth_rate",
        "max_od",
        "exp_phase_start",
        "exp_phase_end",
        "time_at_umax",
        "od_at_umax",
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
    marker_size: int = 5,
    marker_color: str = "red",
    line_color: str = "blue",
    shade_lag="rgba(180,180,180,0.18)",
    shade_exp="rgba(100,149,237,0.16)",
    shade_stat="rgba(144,238,144,0.16)",
    add_phase_shading: bool = True,
    add_window_line: bool = True,
    time_unit: str = "hours",
):
    """
    Draw a single well (points + optional phase shading + optional fit line/curve)
    onto an existing Plotly figure.

    The fit visualization depends on the fit_method in growth stats:
      - "Model Fitting (type)": Displays the fitted growth model curve
      - "Sliding Window": Displays the linear maximum growth rate window

    Works for both:
      - go.Figure() (row/col None)
      - make_subplots() figure (row/col provided)

    Args:
        time_unit: Unit for time display ("seconds", "minutes", or "hours")
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

    # Convert time to display unit
    t_display = convert_hours_to_unit(t, time_unit)
    tmin, tmax = float(t[0]), float(t[-1])
    tmin_display, tmax_display = float(t_display[0]), float(t_display[-1])

    # ---- Phase shading (needs correct xref/yref for each subplot) ----
    bad = is_bad_fit(gs)
    if add_phase_shading and (not bad):
        lag_end = float(np.clip(gs.get("exp_phase_start", tmin), tmin, tmax))
        exp_end = float(np.clip(gs.get("exp_phase_end", tmax), tmin, tmax))
        if exp_end < lag_end:
            exp_end = lag_end

        # Convert to display unit
        lag_end_display = convert_hours_to_unit(lag_end, time_unit)
        exp_end_display = convert_hours_to_unit(exp_end, time_unit)

        # IMPORTANT: xref/yref differ between single-figure and subplots
        if row is None:
            xref = "x"
            yref = "y domain"
        else:
            axis_index = (row - 1) * 12 + col  # for 8x12 plate only
            xref = "x" if axis_index == 1 else f"x{axis_index}"
            yref = "y domain" if axis_index == 1 else f"y{axis_index} domain"

        fig.add_shape(
            type="rect",
            x0=lag_end_display,
            x1=exp_end_display,
            y0=0,
            y1=1,
            xref=xref,
            yref=yref,
            fillcolor=shade_exp,
            line_width=0,
            layer="below",
        )

    # ---- Scatter points ----
    fig.add_trace(
        go.Scatter(
            x=t_display,
            y=y,
            mode="markers",
            marker=dict(size=marker_size, color=marker_color),
            hovertemplate=(
                f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>OD=%{{y:.4f}}<extra></extra>"
            ),
            showlegend=False,
        ),
        **trace_kwargs,
    )

    # ---- Window/gradient line or fitted model curve ----
    if add_window_line:
        # Check which fitting method was used
        fit_method = gs.get("fit_method", "sliding_window")
        is_model_fit = fit_method and "model_fitting" in str(fit_method)

        if is_model_fit:
            # Draw fitted model curve for model-based fits
            model_type = _model_type_from_fit_method(str(fit_method))
            if model_type:

                # Use t_window_start and t_window_end to determine fitting range
                t_win_start = gs.get("t_window_start")
                t_win_end = gs.get("t_window_end")

                # Start with all data, then apply filters
                fit_t = t.copy()
                fit_y = y.copy()

                # Apply window selection if available
                if t_win_start is not None and t_win_end is not None:
                    # Use only the selected time range
                    time_tolerance = 0.1
                    win_mask = (fit_t >= t_win_start - time_tolerance) & (
                        fit_t <= t_win_end + time_tolerance
                    )
                    fit_t = fit_t[win_mask]
                    fit_y = fit_y[win_mask]

                # Refit the model to get the fitted curve (using python_package)
                fit_result = fit_model(fit_t, fit_y, model_type=model_type)

                if fit_result is not None:
                    # Generate dense predictions for smooth curve
                    t_dense = np.linspace(float(fit_t.min()), float(fit_t.max()), 200)
                    t_dense_display = convert_hours_to_unit(t_dense, time_unit)
                    params = fit_result["params"]

                    # python_package models work in linear space directly
                    if fit_result["model_type"] == "richards":
                        y_fit = richards_model(
                            t_dense,
                            params["K"],
                            params["y0"],
                            params["r"],
                            params["t0"],
                            params["nu"],
                        )
                    elif fit_result["model_type"] == "gompertz":
                        y_fit = gompertz_model(
                            t_dense,
                            params["K"],
                            params["y0"],
                            params["mu_max_param"],
                            params["lam"],
                        )
                    else:
                        # Logistic model
                        y_fit = logistic_model(
                            t_dense,
                            params["K"],
                            params["y0"],
                            params["r"],
                            params["t0"],
                        )

                    # Add the fitted curve as a trace
                    fig.add_trace(
                        go.Scatter(
                            x=t_dense_display,
                            y=y_fit,
                            mode="lines",
                            line=dict(width=2, color=line_color),
                            hovertemplate=(
                                f"Model: {model_type}<br>Time=%{{x:.2f}} {time_unit}<br>"
                                f"Fitted OD=%{{y:.4f}}<extra></extra>"
                            ),
                            showlegend=False,
                        ),
                        **trace_kwargs,
                    )
        else:
            # Sliding window: highlight points within the μ_max window in blue
            t_win_start = gs.get("t_window_start")
            t_win_end = gs.get("t_window_end")
            if t_win_start is not None and t_win_end is not None:
                win_mask = (t >= float(t_win_start)) & (t <= float(t_win_end))
                if np.any(win_mask):
                    t_win_display = convert_hours_to_unit(t[win_mask], time_unit)
                    fig.add_trace(
                        go.Scatter(
                            x=t_win_display,
                            y=y[win_mask],
                            mode="markers",
                            marker=dict(size=marker_size + 2, color="blue"),
                            hovertemplate=(
                                f"μ_max window<br>Time=%{{x:.2f}} {time_unit}<br>"
                                f"OD=%{{y:.4f}}<extra></extra>"
                            ),
                            showlegend=False,
                        ),
                        **trace_kwargs,
                    )

    # ---- μ_max point on top ----
    if not is_bad_fit(gs):
        t_umax = gs.get("time_at_umax")
        y_umax = gs.get("od_at_umax")
        if (
            t_umax is not None
            and y_umax is not None
            and np.isfinite(t_umax)
            and np.isfinite(y_umax)
        ):
            t_umax_display = convert_hours_to_unit(float(t_umax), time_unit)
            fig.add_trace(
                go.Scatter(
                    x=[t_umax_display],
                    y=[float(y_umax)],
                    mode="markers",
                    marker=dict(size=marker_size + 7, color="#4CAF50"),
                    hovertemplate=(
                        f"Umax point<br>Time=%{{x:.2f}} {time_unit}<br>"
                        f"OD=%{{y:.4f}}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                **trace_kwargs,
            )


@st.cache_data(show_spinner=False)
def plot_window_single(
    processed_data: dict,
    well: str,
    plot_bgcolor="white",
    paper_bgcolor="white",
    time_unit: str = "hours",
):
    """Plot a single well with lasso selection enabled.

    Args:
        processed_data: Dictionary of processed data by well
        well: Well identifier
        plot_bgcolor: Background color for plot area
        paper_bgcolor: Background color for paper
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
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
        time_unit=time_unit,
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
    fig.update_xaxes(type="linear", showgrid=False, title=get_time_label(time_unit))
    fig.update_yaxes(showgrid=False, title="OD600 (baseline-corrected)")
    return fig


def plot_window_plate(plate: dict, time_unit: str = "hours"):
    """Plot a full 96-well plate overview with window-fit overlays.

    Args:
        plate: Plate dictionary containing processed_data and growth_stats
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    proc = plate.get("processed_data") or {}
    gs_all = plate.get("growth_stats") or {}

    fig = make_subplots(
        rows=8,
        cols=12,
        horizontal_spacing=0.004,
        vertical_spacing=0.02,
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
    # Convert to display unit for range
    x_min_display = convert_hours_to_unit(x_min, time_unit)
    x_max_display = convert_hours_to_unit(x_max, time_unit)
    y_min, y_max = float(min(y.min() for y in ys)), float(max(y.max() for y in ys))
    xr, yr = x_max_display - x_min_display, y_max - y_min
    x_range = [x_min_display - 0.02 * xr, x_max_display + 0.02 * xr]
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
            marker_size=2,  # plate: smaller dots
            time_unit=time_unit,
        )

        # Add well name in top-left corner of each subplot
        axis_suffix = "" if i == 1 else str(i)
        fig.add_annotation(
            text=well,
            xref=f"x{axis_suffix} domain",
            yref=f"y{axis_suffix} domain",
            x=0.05,
            y=0.95,
            showarrow=False,
            font=dict(size=9),
            xanchor="left",
            yanchor="top",
        )

    fig.update_layout(height=900, margin=dict(t=20), showlegend=False)
    fig.update_xaxes(showgrid=False, range=x_range, matches="x")
    fig.update_yaxes(showgrid=False, range=y_range, matches="y")
    return fig


def _vlines(
    fig,
    processed_data: dict,
    well: str,
    *xs,
    gs=None,
    time_unit: str = "hours",
    log_transform: bool = False,
):
    """Add phase shading, phase lines, and fit line/curve annotations to a figure.

    Args:
        fig: Plotly figure to annotate
        processed_data: Dictionary of processed data by well
        well: Well identifier
        *xs: Additional x positions for vertical lines
        gs: Growth statistics dictionary
        time_unit: Unit for time display ("seconds", "minutes", or "hours")
        log_transform: If True, apply ln transformation to all y-values
    """
    # always start clean (important when reusing/copying figures)
    fig.update_layout(shapes=[], annotations=[])
    # Clear existing traces and rebuild with transformed data
    fig.data = []

    # compute range from the real data (NOT from fig.data which may be typed-array dicts)
    d = (processed_data or {}).get(well)
    if d is None or d.empty:
        return fig

    t_raw, y_raw = _finite_sorted_xy(
        d["Time"].to_numpy(), d["baseline_corrected"].to_numpy()
    )
    if t_raw.size == 0:
        return fig

    # Keep raw data for model fitting, apply ln transformation for plotting if requested
    if log_transform:
        # Only keep positive values for log transform
        mask = y_raw > 0
        t = t_raw[mask]
        y = np.log(y_raw[mask])
    else:
        t = t_raw
        y = y_raw

    if t.size == 0:
        return fig

    # Convert time to display unit
    t_display = convert_hours_to_unit(t, time_unit)
    # Use raw time range for boundaries (before any filtering for log transform)
    tmin, tmax = float(t_raw[0]), float(t_raw[-1])
    tmin_display = convert_hours_to_unit(tmin, time_unit)
    tmax_display = convert_hours_to_unit(tmax, time_unit)

    # Determine which points were used in fitting calculations
    y_label = "ln(OD)" if log_transform else "OD"
    gs = gs or {}

    # Check if specific points used in fitting are stored (from lasso selection)
    used_times = gs.get("_used_fit_times")

    if used_times is not None and len(used_times) > 0:
        # Match points by their time values (with tolerance for floating point comparison)
        used_times_arr = np.asarray(used_times)
        time_tolerance = 0.01  # Small tolerance for floating point matching
        used_mask = np.zeros(len(t), dtype=bool)
        for ut in used_times_arr:
            used_mask |= np.abs(t - ut) < time_tolerance

        # Add unused points first (grey)
        unused_mask = ~used_mask
        if np.any(unused_mask):
            fig.add_trace(
                go.Scatter(
                    x=t_display[unused_mask],
                    y=y[unused_mask],
                    mode="markers",
                    marker=dict(size=5, color="grey"),
                    hovertemplate=(
                        f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>{y_label}=%{{y:.4f}}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        # Add used points (red) on top
        if np.any(used_mask):
            fig.add_trace(
                go.Scatter(
                    x=t_display[used_mask],
                    y=y[used_mask],
                    mode="markers",
                    marker=dict(size=5, color="red"),
                    hovertemplate=(
                        f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>{y_label}=%{{y:.4f}}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
    else:
        # No specific points tracked - show all points as red (original behavior)
        fig.add_trace(
            go.Scatter(
                x=t_display,
                y=y,
                mode="markers",
                marker=dict(size=5, color="red"),
                hovertemplate=(
                    f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>{y_label}=%{{y:.4f}}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # --- shading + fit line from growth stats ---
    if gs and not is_bad_fit(gs):
        lag_end = float(np.clip(gs.get("exp_phase_start", tmin), tmin, tmax))
        exp_end = float(np.clip(gs.get("exp_phase_end", tmax), tmin, tmax))
        exp_end = max(exp_end, lag_end)
        max_od = float(gs.get("max_od", 0.0) or 0.0)

        # Transform max_od if log scale
        if log_transform and max_od > 0:
            max_od = np.log(max_od)

        # Convert to display unit
        lag_end_display = convert_hours_to_unit(lag_end, time_unit)
        exp_end_display = convert_hours_to_unit(exp_end, time_unit)

        # colour code exponential phase
        fig.add_vrect(
            x0=lag_end_display,
            x1=exp_end_display,
            fillcolor="rgba(76, 175, 80, 0.22)",
            line_width=0,
            layer="below",
        )

        # add line for max OD600
        if not log_transform or max_od > 0:
            fig.add_hline(
                y=max_od, line=dict(color="rgba(100, 149, 237, 0.6)", width=2)
            )

        # Check which fitting method was used
        fit_method = gs.get("fit_method", "sliding_window")
        is_model_fit = fit_method and "model_fitting" in str(fit_method)

        if is_model_fit:
            # Draw fitted model curve for model-based fits
            model_type = _model_type_from_fit_method(str(fit_method))
            if model_type:

                # Use t_window_start and t_window_end to determine fitting range
                t_win_start = gs.get("t_window_start")
                t_win_end = gs.get("t_window_end")

                # Use raw (non-transformed) data for model fitting
                fit_t = t_raw.copy()
                fit_y = y_raw.copy()

                # Apply window selection if available
                if t_win_start is not None and t_win_end is not None:
                    # Use only the selected time range
                    time_tolerance = 0.1
                    win_mask = (fit_t >= t_win_start - time_tolerance) & (
                        fit_t <= t_win_end + time_tolerance
                    )
                    fit_t = fit_t[win_mask]
                    fit_y = fit_y[win_mask]

                # Refit the model to get the fitted curve (using python_package)
                fit_result = fit_model(fit_t, fit_y, model_type=model_type)

                if fit_result is not None:
                    # Generate dense predictions for smooth curve
                    t_dense = np.linspace(float(fit_t.min()), float(fit_t.max()), 200)
                    t_dense_display = convert_hours_to_unit(t_dense, time_unit)
                    params = fit_result["params"]

                    # python_package models work in linear space directly
                    if fit_result["model_type"] == "richards":
                        y_fit = richards_model(
                            t_dense,
                            params["K"],
                            params["y0"],
                            params["r"],
                            params["t0"],
                            params["nu"],
                        )
                    elif fit_result["model_type"] == "gompertz":
                        y_fit = gompertz_model(
                            t_dense,
                            params["K"],
                            params["y0"],
                            params["mu_max_param"],
                            params["lam"],
                        )
                    else:
                        # Logistic model
                        y_fit = logistic_model(
                            t_dense,
                            params["K"],
                            params["y0"],
                            params["r"],
                            params["t0"],
                        )

                    # Apply ln transformation if requested (transform predictions for display)
                    if log_transform:
                        mask = y_fit > 0
                        t_dense_display = t_dense_display[mask]
                        y_fit = np.log(y_fit[mask])

                    # Add the fitted curve as a trace
                    if len(y_fit) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=t_dense_display,
                                y=y_fit,
                                mode="lines",
                                line=dict(width=2, color="blue"),
                                hovertemplate=(
                                    f"Model: {model_type}<br>Time=%{{x:.2f}} {time_unit}<br>"
                                    f"Fitted {y_label}=%{{y:.4f}}<extra></extra>"
                                ),
                                showlegend=False,
                            )
                        )
        else:
            # Sliding window: highlight points within the μ_max window in blue
            t_win_start = gs.get("t_window_start")
            t_win_end = gs.get("t_window_end")
            if t_win_start is not None and t_win_end is not None:
                win_mask = (t >= float(t_win_start)) & (t <= float(t_win_end))
                if np.any(win_mask):
                    t_win_display = convert_hours_to_unit(t[win_mask], time_unit)
                    fig.add_trace(
                        go.Scatter(
                            x=t_win_display,
                            y=y[win_mask],
                            mode="markers",
                            marker=dict(size=7, color="blue"),
                            hovertemplate=(
                                f"μ_max window<br>Time=%{{x:.2f}} {time_unit}<br>"
                                f"{y_label}=%{{y:.4f}}<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

        # add point at Umax on top of all other traces
        t_umax = gs.get("time_at_umax")
        y_umax = gs.get("od_at_umax")
        if (
            t_umax is not None
            and y_umax is not None
            and np.isfinite(t_umax)
            and np.isfinite(y_umax)
        ):
            y_umax_val = float(y_umax)
            if log_transform:
                if y_umax_val > 0:
                    y_umax_val = np.log(y_umax_val)
                else:
                    y_umax_val = None  # Skip if not positive

            if y_umax_val is not None:
                t_umax_display = convert_hours_to_unit(float(t_umax), time_unit)
                fig.add_trace(
                    go.Scatter(
                        x=[t_umax_display],
                        y=[y_umax_val],
                        mode="markers",
                        marker=dict(size=12, color="#4CAF50"),
                        hovertemplate=(
                            f"Umax point<br>Time=%{{x:.2f}} {time_unit}<br>{y_label}=%{{y:.4f}}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )
    # Constrain axes to the actual data range (prevents infinite lines from extending axes)
    # Add small margin for y-axis for better visualization
    if len(y) > 0:
        y_range = y.max() - y.min()
        y_min = y.min() - 0.05 * y_range
        y_max = y.max() + 0.05 * y_range
    else:
        y_min, y_max = 0, 1

    fig.update_xaxes(range=[tmin_display, tmax_display])
    fig.update_yaxes(range=[y_min, y_max])

    return fig


# --- model-based fits ----------------------------------------------------------
def add_model_fit_well(
    fig,
    *,
    d,  # dataframe for this well (or None/empty)
    well: str,
    gs: dict | None = None,
    row: int | None = None,
    col: int | None = None,
    marker_size: int = 5,
    marker_color: str = "red",
    line_color: str = "blue",
    shade_lag="rgba(180,180,180,0.18)",
    shade_exp="rgba(100,149,237,0.16)",
    shade_stat="rgba(144,238,144,0.16)",
    add_phase_shading: bool = True,
    add_model_curve: bool = True,
    time_unit: str = "hours",
):
    """
    Draw a single well with fitted growth model curve overlay.

    Similar to add_window_well but overlays the fitted parametric model
    instead of the sliding window line.

    Works for both:
      - go.Figure() (row/col None)
      - make_subplots() figure (row/col provided)

    Args:
        time_unit: Unit for time display ("seconds", "minutes", or "hours")
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

    # Convert time to display unit
    t_display = convert_hours_to_unit(t, time_unit)
    tmin, tmax = float(t[0]), float(t[-1])

    # ---- Phase shading ----
    bad = is_bad_fit(gs)
    if add_phase_shading and (not bad):
        lag_end = float(np.clip(gs.get("exp_phase_start", tmin), tmin, tmax))
        exp_end = float(np.clip(gs.get("exp_phase_end", tmax), tmin, tmax))
        if exp_end < lag_end:
            exp_end = lag_end

        # Convert to display unit
        lag_end_display = convert_hours_to_unit(lag_end, time_unit)
        exp_end_display = convert_hours_to_unit(exp_end, time_unit)

        # IMPORTANT: xref/yref differ between single-figure and subplots
        if row is None:
            xref = "x"
            yref = "y domain"
        else:
            axis_index = (row - 1) * 12 + col  # for 8x12 plate only
            xref = "x" if axis_index == 1 else f"x{axis_index}"
            yref = "y domain" if axis_index == 1 else f"y{axis_index} domain"

        fig.add_shape(
            type="rect",
            x0=lag_end_display,
            x1=exp_end_display,
            y0=0,
            y1=1,
            xref=xref,
            yref=yref,
            fillcolor=shade_exp,
            line_width=0,
            layer="below",
        )

    # ---- Scatter points ----
    fig.add_trace(
        go.Scatter(
            x=t_display,
            y=y,
            mode="markers",
            marker=dict(size=marker_size, color=marker_color),
            hovertemplate=(
                f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>OD=%{{y:.4f}}<extra></extra>"
            ),
            showlegend=False,
        ),
        **trace_kwargs,
    )

    # Note: Model fit curve overlay is now handled by _vlines() function
    # to consolidate all growth line/curve drawing logic in one place


@st.cache_data(show_spinner=False)
def plot_model_fit_single(
    processed_data: dict,
    growth_stats: dict,
    well: str,
    plot_bgcolor="white",
    paper_bgcolor="white",
    time_unit: str = "hours",
):
    """Plot a single well with fitted growth model overlay and lasso selection enabled.

    Args:
        processed_data: Dictionary of processed data by well
        growth_stats: Dictionary of growth statistics by well
        well: Well identifier
        plot_bgcolor: Background color for plot area
        paper_bgcolor: Background color for paper
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    d = (processed_data or {}).get(well)
    gs = (growth_stats or {}).get(well)

    fig = go.Figure()

    add_model_fit_well(
        fig,
        d=d,
        well=well,
        gs=gs,
        marker_size=5,
        marker_color="red",
        line_color="blue",
        time_unit=time_unit,
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        uirevision="keep",
        dragmode="lasso",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_xaxes(type="linear", showgrid=False, title=get_time_label(time_unit))
    fig.update_yaxes(showgrid=False, title="OD600 (baseline-corrected)")

    return fig


@st.cache_data(show_spinner=False)
def plot_model_fit_single_annotated(
    d,
    gs: dict,
    well: str,
    plot_bgcolor="white",
    paper_bgcolor="white",
    time_unit: str = "hours",
):
    """
    Plot a single well with model fit and annotations for phase boundaries.
    Similar to plot_window_single_annotated but for model-based fits.

    Args:
        d: DataFrame with Time and baseline_corrected columns
        gs: Growth statistics dictionary
        well: Well identifier
        plot_bgcolor: Background color for plot area
        paper_bgcolor: Background color for paper
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
    """
    time_label = get_time_label(time_unit)
    if d is None or d.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Well {well} - No data",
            xaxis_title=time_label,
            yaxis_title="OD600",
        )
        return fig

    fig = go.Figure()

    t, y = _finite_sorted_xy(d["Time"].to_numpy(), d["baseline_corrected"].to_numpy())
    if t.size == 0:
        return fig

    # Convert time to display unit
    t_display = convert_hours_to_unit(t, time_unit)
    tmin, tmax = float(t[0]), float(t[-1])

    # --- shading + fit curve from growth stats ---
    gs = gs or {}
    if gs and not is_bad_fit(gs):
        lag_end = float(np.clip(gs.get("exp_phase_start", tmin), tmin, tmax))
        exp_end = float(np.clip(gs.get("exp_phase_end", tmax), tmin, tmax))
        exp_end = max(exp_end, lag_end)
        max_od = float(gs.get("max_od", 0.0) or 0.0)

        # Convert to display unit
        lag_end_display = convert_hours_to_unit(lag_end, time_unit)
        exp_end_display = convert_hours_to_unit(exp_end, time_unit)

        # add line for lag end
        fig.add_vline(x=lag_end_display, line_dash="dot")

        # colour code exponential phase
        fig.add_vrect(
            x0=lag_end_display,
            x1=exp_end_display,
            fillcolor="rgba(100,149,237,0.16)",
            line_width=0,
            layer="below",
        )

        # add line for exp end
        fig.add_vline(x=exp_end_display, line_dash="dot")

        # add line for max OD600
        fig.add_hline(y=max_od, line_dash="dot")

        # Add fitted model curve
        fit_method = gs.get("fit_method", "")
        if fit_method and "model_fitting" in fit_method:
            # Extract model type from fit_method string (e.g., "model_fitting_logistic")
            model_type = fit_method.replace("model_fitting_", "")

            # Refit the model to get the fitted curve (using python_package)
            fit_result = fit_model(t, y, model_type=model_type)

            if fit_result is not None:
                t_dense = np.linspace(float(t.min()), float(t.max()), 200)
                t_dense_display = convert_hours_to_unit(t_dense, time_unit)
                params = fit_result["params"]

                # python_package models work in linear space directly
                if fit_result["model_type"] == "richards":
                    y_fit = richards_model(
                        t_dense,
                        params["K"],
                        params["y0"],
                        params["r"],
                        params["t0"],
                        params["nu"],
                    )
                elif fit_result["model_type"] == "gompertz":
                    y_fit = gompertz_model(
                        t_dense,
                        params["K"],
                        params["y0"],
                        params["mu_max_param"],
                        params["lam"],
                    )
                else:
                    # Logistic model
                    y_fit = logistic_model(
                        t_dense, params["K"], params["y0"], params["r"], params["t0"]
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=t_dense_display,
                            y=y_fit,
                            mode="lines",
                            line=dict(width=3, color="rgba(30, 144, 255, 0.7)"),
                            name=f"Fitted {model_type} model",
                            showlegend=True,
                        )
                    )

    # Add data points
    fig.add_trace(
        go.Scatter(
            x=t_display,
            y=y,
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Data",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=f"Well {well} - Model Fit",
        xaxis_title=time_label,
        yaxis_title="OD600 (baseline corrected)",
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        hovermode="closest",
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


def plot_window_single_d1(
    plate: dict,
    well: str,
    sg_window=11,
    sg_poly=2,
    frac_peak=0.15,
    add_fit=True,
    time_unit: str = "hours",
    gs: dict | None = None,
):
    """Plot the first derivative of a well's smoothed curve.

    Args:
        plate: Plate dictionary containing processed_data
        well: Well identifier
        sg_window: Savitzky-Golay window size
        sg_poly: Savitzky-Golay polynomial order
        frac_peak: Fraction of peak for threshold
        add_fit: Whether to add fitted curve
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
        gs: Growth statistics dictionary (optional). If provided with _used_fit_times,
            only the lasso-selected data points will be used for derivative calculation.
    """
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t_full = d["Time"].to_numpy(float)
    y_full = d["baseline_corrected"].to_numpy(float)

    # Store full time range for x-axis before any filtering
    t_full_display = convert_hours_to_unit(t_full, time_unit)
    x_range = [float(t_full_display.min()), float(t_full_display.max())]

    t = t_full.copy()
    y = y_full.copy()

    # Filter to lasso-selected points if available
    gs = gs or {}
    used_times = gs.get("_used_fit_times")
    if used_times is not None and len(used_times) > 0:
        used_times_arr = np.asarray(used_times)
        time_tolerance = 0.01  # Small tolerance for floating point matching
        used_mask = np.zeros(len(t), dtype=bool)
        for ut in used_times_arr:
            used_mask |= np.abs(t - ut) < time_tolerance
        t = t[used_mask]
        y = y[used_mask]

    if len(t) < 3:
        return go.Figure()

    # Apply smoothing before computing derivative
    y_s = smooth(y, sg_window, sg_poly)

    # Compute first derivative using the data processing function
    t, dy = compute_first_derivative(t, y_s)

    # Convert time to display unit
    t_display = convert_hours_to_unit(t, time_unit)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_display,
            y=dy,
            mode="lines",
            line=dict(width=2),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>dy/dt=%{{y:.6f}}<extra></extra>",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    if add_fit:
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
                    x=t_display,
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
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, title=get_time_label(time_unit), range=x_range)
    fig.update_yaxes(showgrid=False, title="d(OD)/dt")
    return fig


def plot_window_single_d2(
    plate: dict,
    well: str,
    sg_window=11,
    sg_poly=2,
    add_fit=True,
    time_unit: str = "hours",
    gs: dict | None = None,
):
    """Plot the second derivative of a well's smoothed curve.

    Args:
        plate: Plate dictionary containing processed_data
        well: Well identifier
        sg_window: Savitzky-Golay window size
        sg_poly: Savitzky-Golay polynomial order
        add_fit: Whether to add fitted curve
        time_unit: Unit for time axis display ("seconds", "minutes", or "hours")
        gs: Growth statistics dictionary (optional). If provided with _used_fit_times,
            only the lasso-selected data points will be used for derivative calculation.
    """
    d = (plate.get("processed_data") or {}).get(well)
    if d is None or d.empty:
        return go.Figure()

    t_full = d["Time"].to_numpy(float)
    y_full = d["baseline_corrected"].to_numpy(float)

    # Store full time range for x-axis before any filtering
    t_full_display = convert_hours_to_unit(t_full, time_unit)
    x_range = [float(t_full_display.min()), float(t_full_display.max())]

    t = t_full.copy()
    y = y_full.copy()

    # Filter to lasso-selected points if available
    gs = gs or {}
    used_times = gs.get("_used_fit_times")
    if used_times is not None and len(used_times) > 0:
        used_times_arr = np.asarray(used_times)
        time_tolerance = 0.01  # Small tolerance for floating point matching
        used_mask = np.zeros(len(t), dtype=bool)
        for ut in used_times_arr:
            used_mask |= np.abs(t - ut) < time_tolerance
        t = t[used_mask]
        y = y[used_mask]

    if len(t) < 3:
        return go.Figure()

    # Apply smoothing before computing derivative
    y_s = smooth(y, sg_window, sg_poly)

    # Compute second derivative using the data processing function
    t, d2y = compute_second_derivative(t, y_s)

    # Convert time to display unit
    t_display = convert_hours_to_unit(t, time_unit)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_display,
            y=d2y,
            mode="lines",
            line=dict(width=2),
            hovertemplate=f"Well={well}<br>Time=%{{x:.2f}} {time_unit}<br>d²y/dt²=%{{y:.6f}}<extra></extra>",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if add_fit:
        # Need to compute first derivative for fitting
        _, dy = compute_first_derivative(t, y_s)
        popt = _fit_idealised_derivatives(t, dy)
        d2_fit = None
        if popt is not None:
            A, r, t0 = popt
            d2_fit = d2_model(t, A, r, t0)

        if d2_fit is not None and np.isfinite(d2_fit).any():
            fig.add_trace(
                go.Scatter(
                    x=t_display,
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
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, title=get_time_label(time_unit), range=x_range)
    fig.update_yaxes(showgrid=False, title="d²(OD)/dt²")
    return fig


def plot_rmse_heatmap(plate: dict):
    """
    Plot a 96-well plate heatmap of RMSE values.

    The heatmap is centered on 0 (green) with red indicating higher RMSE values.
    Applicable for both sliding window and model-based fits.

    Args:
        plate: Plate dictionary containing growth_stats

    Returns:
        Plotly figure with RMSE heatmap
    """
    growth_stats = plate.get("growth_stats") or {}

    # Create a 8x12 grid for the plate layout
    rows = "ABCDEFGH"
    cols = range(1, 13)

    # Extract RMSE values and organize into plate layout
    rmse_matrix = []
    hover_text = []
    well_labels = []

    for row in rows:
        rmse_row = []
        hover_row = []
        label_row = []
        for col in cols:
            well = f"{row}{col}"
            gs = growth_stats.get(well, {})
            rmse = gs.get("model_rmse", np.nan)

            rmse_row.append(rmse if pd.notna(rmse) else np.nan)
            hover_row.append(
                f"Well: {well}<br>RMSE: {rmse:.5f}"
                if pd.notna(rmse)
                else f"Well: {well}<br>RMSE: N/A"
            )
            label_row.append(well)

        rmse_matrix.append(rmse_row)
        hover_text.append(hover_row)
        well_labels.append(label_row)

    # Convert to numpy array for easier manipulation
    rmse_matrix = np.array(rmse_matrix)

    # Find the maximum absolute RMSE for symmetric color scale
    finite_rmse = rmse_matrix[np.isfinite(rmse_matrix)]
    if len(finite_rmse) == 0:
        max_rmse = 0.1
    else:
        max_rmse = np.max(np.abs(finite_rmse))

    # Create the heatmap with colorblind-friendly colors
    # Using a teal-white-orange scale that's accessible to most types of colorblindness
    fig = go.Figure(
        data=go.Heatmap(
            z=rmse_matrix,
            x=[str(c) for c in cols],
            y=list(rows),
            colorscale=[
                [0.0, "rgb(68, 170, 153)"],  # Teal/cyan at 0 (good fit)
                [0.5, "rgb(238, 238, 238)"],  # Light gray at midpoint
                [1.0, "rgb(221, 132, 82)"],  # Orange at max (poor fit)
            ],
            zmid=0,  # Center the color scale at 0
            zmin=0,
            zmax=max_rmse if max_rmse > 0 else 0.1,
            text=well_labels,
            texttemplate="%{text}",
            textfont=dict(size=10, color="black"),
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            showscale=False,  # Remove the colorbar legend
            xgap=1,  # Add gap between cells (creates black outline effect)
            ygap=1,
        )
    )

    fig.update_layout(
        title="Model Fit Quality (RMSE)",
        xaxis=dict(
            visible=False,  # Hide x-axis
        ),
        yaxis=dict(
            visible=False,  # Hide y-axis
            autorange="reversed",  # Reverse y-axis so A1 is at top left
        ),
        width=800,
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",  # White background
        paper_bgcolor="white",  # White paper background
    )

    return fig
