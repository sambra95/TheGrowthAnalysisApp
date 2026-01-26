"""Upload inputs, configure analysis parameters, and run plate analysis."""

import pandas as pd
import streamlit as st

from functions.data_processing import analyse_plate, load_plate

ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))

GREEN = "🟩"
ORANGE = "🟧"
RED = "🟥"
BLUE = "🟦"
GRAY = "⬜"

DEFAULT_PARAMS = dict(
    time_unit="minutes",  # "seconds", "minutes", or "hours"
    pathlength_cm_=0.42,
    clip_time_series=(0.0, 72.0),
    remove_wells=False,
    blank=True,
    window_points=15,
    lag_cutoff=0.1,
    exp_cutoff=0.1,
    sg_window=15,
    sg_poly=2,
    min_data_points=5,
    min_signal_to_noise=5.0,
    min_growth_rate=0.001,
    growth_method="Sliding Window",
    model_type="logistic",
)


def init_state():
    """Ensure required session state keys exist."""
    st.session_state.setdefault("plates", {})
    return st.session_state


def plate_params(ss, plate_id: str) -> dict:
    """Return stored params for a plate or the defaults."""
    return (ss.plates.get(plate_id, {}) or {}).get("params", DEFAULT_PARAMS)


def build_symbol_grid(
    *, plate_map: pd.DataFrame, present: set[str], remove_wells=False, blank=True
):
    """Build a grid of well status symbols for the plate preview."""
    removed = {w.upper() for w in remove_wells} if remove_wells else set()
    name_by_well = {f"{r}{c}": str(plate_map.loc[r, c]) for r in ROWS for c in COLS}
    ignored = {w for w, nm in name_by_well.items() if nm == "False"}

    grid = pd.DataFrame(index=ROWS, columns=COLS, dtype="object")
    for r in ROWS:
        for c in COLS:
            w = f"{r}{c}"
            nm = name_by_well.get(w, "")
            is_blank = nm == "BLANK"

            if w in removed:
                sym = RED
            elif w in present:
                sym = GREEN
            elif blank and is_blank:
                sym = BLUE
            elif w in ignored:
                sym = GRAY
            else:
                sym = ORANGE

            grid.loc[r, c] = sym
    return grid


def render_plate_table(grid: pd.DataFrame):
    """Render an HTML table showing the plate status grid."""
    css = """
    <style>
      .plate-wrap { width: 100%; overflow: hidden; }
      table.plate {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 18px;
      }
      table.plate th, table.plate td {
        border: 1px solid rgba(49,51,63,0.2);
        text-align: center;
        padding: 6px 0;
        line-height: 1.2;
      }
      table.plate th { font-weight: 600; }
      table.plate th.row { width: 2.2rem; }
    </style>
    """

    header = "".join(f"<th>{c}</th>" for c in grid.columns)
    rows_html = []
    for r in grid.index:
        cells = "".join(f"<td>{grid.loc[r, c]}</td>" for c in grid.columns)
        rows_html.append(f"<tr><th class='row'>{r}</th>{cells}</tr>")

    html = f"""
    {css}
    <div class="plate-wrap">
      <table class="plate">
        <thead>
          <tr><th></th>{header}</tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# ---------------- App ----------------
ss = init_state()

title_col, popover_col = st.columns([9, 2])
with title_col:
    st.title("Upload and Analyze")
with popover_col:
    st.write("")
    with st.popover("Explain this page to me", width="stretch"):
        st.markdown(
            """
**Actions you can perform on this page:**
- Upload data files. This excel table contains the time-series data for your growth curves, arranged with each column corresponding to a well
- Upload plate maps. This excel table dictates how each sample will be named in the app.
- Configure analysis parameters (read interval, pathlength, time range, etc.)
- Preview your plate layout before analysis
- Run growth curve analysis on the uploaded plates

"""
        )

st.divider()

# Upload (store bytes directly into ss.plates[plate_id]).
u1, u2 = st.columns(2)
with u1:
    with st.container(border=True):
        header_col, popover_col = st.columns([0.85, 0.15])
        with header_col:
            st.header("Step 1. Upload data file")
        with popover_col:
            with st.popover("Help", width="stretch"):
                st.markdown("**Required Data File Format:**")
                st.markdown("Excel file (.xlsx or .xls) with time series data")
                st.info(
                    "Your data file must include a **Time** column with numeric values (integers or decimals). "
                    "Select the time unit (seconds, minutes, or hours) in Step 3."
                )

                # Create example data table
                example_data = pd.DataFrame(
                    {
                        "Time": [0, 12, 24, 36],
                        "A1": [0.05, 0.08, 0.15, 0.28],
                        "A2": [0.06, 0.09, 0.18, 0.32],
                        "B1": [0.05, 0.07, 0.14, 0.26],
                        "...": ["...", "...", "...", "..."],
                    }
                )
                st.dataframe(example_data, hide_index=True, width="stretch")

                # Download example file
                with open("example_data.xlsx", "rb") as f:
                    st.download_button(
                        "Download example data",
                        data=f.read(),
                        file_name="example_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width="stretch",
                        type="primary",
                    )
        data_file = st.file_uploader(
            "Plate reader Excel (.xlsx/.xls)", ["xlsx", "xls"], key="data_up"
        )
with u2:
    with st.container(border=True):
        header_col, popover_col = st.columns([0.85, 0.15])
        with header_col:
            st.header("Step 2. Upload plate map")
        with popover_col:
            with st.popover("Help", width="stretch"):
                st.markdown("**Required Plate Map Format:**")
                st.markdown("Excel file (.xlsx or .xls) with sample layout")
                st.markdown(
                    """
                    - 96-well plate format (rows A-H, columns 1-12)
                    - Samples with the same name will be assigned as replicates
                    - Use 'BLANK' for blank wells
                    - Leave cells empty for wells to ignore
                    - The first '_' is used to split strain and condition labels. These can be used to group samples and colour code with a legend in the 'Create Visulations' page.
                    """
                )

                # Create example plate map table
                example_map = pd.DataFrame(
                    {
                        "rows": ["A", "B", "C", "D"],
                        "1": [
                            "Sample1_Condition1",
                            "Sample3_Condition2",
                            "",
                            "Sample6_Condition3",
                        ],
                        "2": [
                            "Sample1_Condition2",
                            "BLANK",
                            "Sample5_Condition2",
                            "Sample7_Condition2",
                        ],
                        "3": [
                            "Sample2_Condition1",
                            "Sample4_Condition3",
                            "Sample5_Condition2",
                            "BLANK",
                        ],
                        "...": ["...", "...", "...", "..."],
                    }
                )
                st.dataframe(example_map, hide_index=True, width="stretch")

                # Download example file
                with open("example_plate_map.xls", "rb") as f:
                    st.download_button(
                        "Download example plate map",
                        data=f.read(),
                        file_name="example_plate_map.xls",
                        mime="application/vnd.ms-excel",
                        width="stretch",
                        type="primary",
                    )
        map_file = st.file_uploader(
            "Plate map (.xls/.xlsx) with 'rows' column", ["xlsx", "xls"], key="map_up"
        )

if st.button(
    "Load plate",
    type="primary",
    width="stretch",
    disabled=not (data_file and map_file),
):
    plate_id = (
        data_file.name.rsplit(".", 1)[0]
        if getattr(data_file, "name", None)
        else "Plate"
    )
    load_plate(
        ss.plates,
        plate_id,
        data_bytes=data_file.getvalue(),
        plate_bytes=map_file.getvalue(),
        params=DEFAULT_PARAMS,
    )
    st.toast(f"Saved uploads for {plate_id}")


ready = sorted(ss.plates)

# Step 3: Select plate and input metadata
with st.container(border=True):
    st.header("Step 3. Select plate and preprocessing parameters")

    pcol, acol = st.columns(2, gap="large")

    with pcol:
        plate_id = st.selectbox("Plate to analyse", ready, disabled=not ready)
        params0 = plate_params(ss, plate_id) if plate_id else DEFAULT_PARAMS

        a, b, c = st.columns(3, vertical_alignment="center")
        time_unit = a.selectbox(
            "Time unit in data file",
            options=["seconds", "minutes", "hours"],
            index=["seconds", "minutes", "hours"].index(
                params0.get("time_unit", "hours")
            ),
            help="Select the unit of time values in your data file's Time column",
        )
        pl_cm = b.number_input(
            "Pathlength (cm)",
            value=float(params0["pathlength_cm_"]),
            step=0.01,
            format="%.3f",
            help="Optical pathlength of the plate reader (used to normalize OD600 values to 1 cm pathlength)",
        )
        blank = c.checkbox(
            "Blank subtraction (label 'BLANK')",
            bool(params0["blank"]),
            help="Subtract the mean of all wells labeled 'BLANK' as baseline correction",
        )

        a, b = st.columns(2)
        clip_time_series = (
            float(
                a.number_input(
                    "Start (h)",
                    0.0,
                    1e6,
                    float(params0["clip_time_series"][0]),
                    0.5,
                    help="Starting time for analysis (earlier time points will be excluded)",
                )
            ),
            float(
                b.number_input(
                    "End (h)",
                    0.0,
                    1e6,
                    float(params0["clip_time_series"][1]),
                    0.5,
                    help="Ending time for analysis (later time points will be excluded)",
                )
            ),
        )

        # Get default excluded wells from params0
        default_excluded = params0.get("remove_wells", [])
        if default_excluded is False or not default_excluded:
            default_excluded = []

        remove_wells = st.multiselect(
            "Exclude wells",
            options=[f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)],
            default=default_excluded,
            help="Manually exclude specific wells from analysis (e.g., contaminated samples)",
        )

        # Preserve the False sentinel behavior used elsewhere.
        remove_wells = remove_wells if remove_wells else False

        st.write("")
        if st.button(
            "Remove selected plate",
            type="tertiary",
            width="stretch",
            disabled=not plate_id,
        ):
            ss.plates.pop(plate_id, None)
            st.rerun()

    with acol:
        # Preview grid in Step 3
        if plate_id:
            rec = ss.plates.get(plate_id, {})
            if rec.get("uploads"):
                # Build a temporary params dict for preview (will be completed in Step 4)
                preview_params = dict(
                    time_unit=str(time_unit),
                    pathlength_cm_=float(pl_cm),
                    clip_time_series=clip_time_series,
                    remove_wells=remove_wells,
                    blank=bool(blank),
                    window_points=int(params0.get("window_points", 15)),
                    lag_cutoff=float(params0.get("lag_cutoff", 0.1)),
                    exp_cutoff=float(params0.get("exp_cutoff", 0.1)),
                    sg_window=int(params0.get("sg_window", 15)),
                    sg_poly=int(params0.get("sg_poly", 2)),
                    min_data_points=int(params0.get("min_data_points", 5)),
                    min_signal_to_noise=float(params0.get("min_signal_to_noise", 5.0)),
                    min_growth_rate=float(params0.get("min_growth_rate", 0.001)),
                    growth_method=str(params0.get("growth_method", "Sliding Window")),
                    model_type=str(params0.get("model_type", "logistic")),
                )
                tmp = {"uploads": rec["uploads"], "params": preview_params}
                plate_preview = analyse_plate(tmp)
                present = set(plate_preview.get("growth_stats", {}).keys())

                grid = build_symbol_grid(
                    plate_map=plate_preview["plate_map"],
                    present=present,
                    remove_wells=preview_params["remove_wells"],
                    blank=preview_params["blank"],
                )

                st.subheader(plate_id)
                st.caption(
                    "· 🟩 sample · 🟦 blank · 🟥 excluded by user · 🟧 not in data file · ⬜ not in plate map"
                )
                render_plate_table(grid)

        else:
            st.warning("Upload files to see plate preview.")

# Step 4: Select analysis parameters
with st.container(border=True):
    st.header("Step 4. Select the analysis parameters")

    header_col, help_col = st.columns([0.85, 0.15])
    with header_col:
        st.markdown("**Growth Descriptor Calculation Method**")
    with help_col:
        with st.popover("Help", width="stretch"):
            import numpy as np
            import plotly.graph_objects as go

            st.markdown("### Growth Descriptor Calculation Methods")

            # Sliding Window Expander
            with st.expander("Sliding Window Method", expanded=False):
                # Generate data points (scatter)
                t_points = np.linspace(0, 48, 50)
                y_points = 0.05 + 0.95 / (1 + np.exp(-0.2 * (t_points - 24)))

                fig_sw = go.Figure()

                # Add the data as scatter points
                fig_sw.add_trace(
                    go.Scatter(
                        x=t_points,
                        y=y_points,
                        mode="markers",
                        marker=dict(color="blue", size=6),
                        name="Data points",
                    )
                )

                # Add single sliding window box
                window_center_t = 24
                window_half_width = 4
                win_x0 = window_center_t - window_half_width
                win_x1 = window_center_t + window_half_width

                # Get y values at window boundaries for the box
                y_at_win = 0.05 + 0.95 / (
                    1 + np.exp(-0.2 * (np.array([win_x0, win_x1]) - 24))
                )
                box_y_min = min(y_at_win) - 0.08
                box_y_max = max(y_at_win) + 0.08

                # Draw the sliding window box
                fig_sw.add_shape(
                    type="rect",
                    x0=win_x0,
                    x1=win_x1,
                    y0=box_y_min,
                    y1=box_y_max,
                    fillcolor="rgba(0,200,0,0.2)",
                    line=dict(color="green", width=2),
                )

                # Add "Sliding Window" label
                fig_sw.add_annotation(
                    x=window_center_t,
                    y=box_y_max + 0.08,
                    text="Sliding Window",
                    showarrow=False,
                    font=dict(color="green", size=12),
                )

                # Add dashed arrows showing movement direction
                arrow_y = box_y_min + (box_y_max - box_y_min) / 2

                # Left arrow (showing it came from left)
                fig_sw.add_annotation(
                    x=win_x0 - 1,
                    y=arrow_y,
                    ax=win_x0 - 6,
                    ay=arrow_y,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="gray",
                    text="",
                )

                # Right arrow (showing it moves to right)
                fig_sw.add_annotation(
                    x=win_x1 + 6,
                    y=arrow_y,
                    ax=win_x1 + 1,
                    ay=arrow_y,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="gray",
                    text="",
                )

                fig_sw.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Time (h)",
                    yaxis_title="OD600",
                    showlegend=False,
                )
                st.plotly_chart(fig_sw, width="stretch", config={"staticPlot": True})

                st.markdown(
                    """
**How it works:**
1. A window of fixed size (e.g., 15 points) slides across the growth curve
2. At each position, a linear regression is fitted to log-transformed OD values
3. The slope of each fit represents the growth rate (μ) at that window
4. The **maximum slope** across all windows is reported as **μ_max**

**Output metrics:**
- **μ_max**: Maximum specific growth rate (h⁻¹)
- **Doubling time**: ln(2) / μ_max
- **Lag time**: Time at end of lag phase
                    """
                )

            # Model Fitting Expander
            with st.expander("Model Fitting Method", expanded=False):
                st.markdown(
                    """
**How it works:**
1. A parametric growth model is fitted to the entire growth curve
2. The model's analytical derivative gives the growth rate at each time point
3. **μ_max** is the maximum of the derivative (occurs at the inflection point)

**Available models:**
                    """
                )

                t = np.linspace(0, 48, 200)

                # Logistic model
                st.markdown("**Logistic**")
                st.latex(r"y(t) = \frac{A}{1 + e^{-\mu(t - \lambda)}}")
                st.caption(
                    "Classic S-shaped curve with symmetric inflection point. Most commonly used for microbial growth."
                )
                y_logistic = 1.0 / (1 + np.exp(-0.15 * (t - 24)))
                fig_log = go.Figure()
                fig_log.add_trace(
                    go.Scatter(
                        x=t,
                        y=y_logistic,
                        mode="lines",
                        line=dict(color="blue", width=2),
                    )
                )
                fig_log.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Time (h)",
                    yaxis_title="OD600",
                    showlegend=False,
                )
                st.plotly_chart(fig_log, width="stretch", config={"staticPlot": True})

                # Gompertz model
                st.markdown("**Gompertz**")
                st.latex(r"y(t) = A \cdot e^{-e^{-\mu(t - \lambda)}}")
                st.caption(
                    "Asymmetric S-curve with slower approach to stationary phase. Often fits bacterial growth better than logistic."
                )
                y_gompertz = 1.0 * np.exp(-np.exp(-0.15 * (t - 24)))
                fig_gom = go.Figure()
                fig_gom.add_trace(
                    go.Scatter(
                        x=t,
                        y=y_gompertz,
                        mode="lines",
                        line=dict(color="green", width=2),
                    )
                )
                fig_gom.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Time (h)",
                    yaxis_title="OD600",
                    showlegend=False,
                )
                st.plotly_chart(fig_gom, width="stretch", config={"staticPlot": True})

                # Richards model
                st.markdown("**Richards**")
                st.latex(
                    r"y(t) = \frac{A}{(1 + \nu \cdot e^{-\mu(t - \lambda)})^{1/\nu}}"
                )
                st.caption(
                    "Generalized logistic with shape parameter ν. Most flexible - use when other models don't fit well."
                )
                nu = 2.0
                y_richards = 1.0 / (1 + nu * np.exp(-0.15 * (t - 24))) ** (1 / nu)
                fig_ric = go.Figure()
                fig_ric.add_trace(
                    go.Scatter(
                        x=t,
                        y=y_richards,
                        mode="lines",
                        line=dict(color="orange", width=2),
                    )
                )
                fig_ric.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Time (h)",
                    yaxis_title="OD600",
                    showlegend=False,
                )
                st.plotly_chart(fig_ric, width="stretch", config={"staticPlot": True})

            # Phase Boundary Detection Expander
            with st.expander("Phase Boundary Detection", expanded=False):
                st.markdown(
                    """
**Both methods use the same approach** for detecting phase boundaries, based on the first derivative (growth rate) of the curve:

1. **Lag phase end**: First time point where growth rate exceeds the threshold
2. **Exponential phase end**: First time point *after* peak where rate drops below threshold

**The threshold** is set as a fraction of μ_max (default 10%).
                    """
                )

                # Generate derivative curve data for illustration
                t_deriv = np.linspace(0, 48, 200)
                exp_term = np.exp(-0.2 * (t_deriv - 24))
                y_deriv = 0.95 * 0.2 * exp_term / (1 + exp_term) ** 2

                mu_max = np.max(y_deriv)
                t_max = t_deriv[np.argmax(y_deriv)]
                sigma = 8
                y_fitted = mu_max * np.exp(-((t_deriv - t_max) ** 2) / (2 * sigma**2))

                illustration_threshold = 0.25 * mu_max
                left_crossings = np.where(y_fitted[:100] >= illustration_threshold)[0]
                t_lag_end = (
                    t_deriv[left_crossings[0]] if len(left_crossings) > 0 else 10
                )
                right_crossings = np.where(y_fitted[100:] >= illustration_threshold)[0]
                t_exp_end = (
                    t_deriv[100 + right_crossings[-1]]
                    if len(right_crossings) > 0
                    else 38
                )

                fig_deriv = go.Figure()

                t_deriv_points = np.linspace(0, 48, 50)
                exp_term_pts = np.exp(-0.2 * (t_deriv_points - 24))
                y_deriv_points = 0.95 * 0.2 * exp_term_pts / (1 + exp_term_pts) ** 2

                fig_deriv.add_trace(
                    go.Scatter(
                        x=t_deriv_points,
                        y=y_deriv_points,
                        mode="markers",
                        marker=dict(color="blue", size=6),
                        name="dOD/dt",
                    )
                )

                fig_deriv.add_trace(
                    go.Scatter(
                        x=t_deriv,
                        y=y_fitted,
                        mode="lines",
                        line=dict(color="orange", width=2, dash="dash"),
                        name="Fitted curve",
                    )
                )

                threshold = 0.25 * mu_max
                fig_deriv.add_hline(
                    y=threshold,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Cutoff",
                    annotation_position="right",
                )

                fig_deriv.add_vline(
                    x=t_lag_end,
                    line_dash="solid",
                    line_color="green",
                    line_width=2,
                )
                fig_deriv.add_vline(
                    x=t_exp_end,
                    line_dash="solid",
                    line_color="red",
                    line_width=2,
                )

                fig_deriv.add_annotation(
                    x=t_lag_end,
                    y=mu_max * 0.9,
                    text="End of<br>Lag Phase",
                    showarrow=False,
                    font=dict(color="green", size=10),
                    xanchor="right",
                    xshift=-5,
                )
                fig_deriv.add_annotation(
                    x=t_exp_end,
                    y=mu_max * 0.9,
                    text="End of<br>Exp. Phase",
                    showarrow=False,
                    font=dict(color="red", size=10),
                    xanchor="left",
                    xshift=5,
                )

                fig_deriv.add_annotation(
                    x=t_max,
                    y=mu_max + 0.003,
                    text="μ_max",
                    showarrow=False,
                    font=dict(color="orange", size=11),
                )

                fig_deriv.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Time (h)",
                    yaxis_title="dOD/dt (h⁻¹)",
                    showlegend=False,
                )
                st.plotly_chart(fig_deriv, width="stretch", config={"staticPlot": True})

                st.markdown(
                    """
**Difference between methods:**

| | Sliding Window | Model Fitting |
|---|---|---|
| **Derivative source** | Savitzky-Golay smoothed raw data | Analytical derivative of fitted model |
| **Curve used** | Smoothed empirical data | Parametric model prediction |

**Threshold parameters:**
- *Lag cutoff*: Fraction of μ_max for lag phase end detection (default 10%)
- *Exp cutoff*: Fraction of μ_max for exponential phase end detection (default 10%)

Lower values detect transitions earlier; higher values require more pronounced rate changes.
                    """
                )

    method_col, option_col = st.columns(2)
    with method_col:
        growth_method = st.selectbox(
            "Method",
            options=["Sliding Window", "Model Fitting"],
            index=(
                0
                if params0.get("growth_method", "Sliding Window") == "Sliding Window"
                else 1
            ),
            help="Choose how to calculate growth descriptors: Sliding Window uses linear fits over a moving window, Model Fitting fits parametric growth curves",
        )
    with option_col:
        # Method-specific options
        if growth_method == "Sliding Window":
            window_points = st.number_input(
                "Window size (points)",
                5,
                200,
                int(params0["window_points"]),
                1,
                help="Number of consecutive data points used for sliding window linear fit to determine maximum growth rate",
            )
            model_type = params0.get("model_type", "logistic")
        else:
            # Model Fitting selected
            model_type = st.selectbox(
                "Growth model",
                options=["logistic", "gompertz", "richards"],
                index=["logistic", "gompertz", "richards"].index(
                    params0.get("model_type", "logistic")
                ),
                help="Parametric model to fit to the growth curve",
            )
            window_points = int(params0["window_points"])

    # Phase boundary cutoffs - apply to both Sliding Window and Model Fitting methods
    st.write("")
    st.markdown("**Phase Boundary Detection**")
    lag_col, exp_col = st.columns(2)
    with lag_col:
        lag_cutoff = st.number_input(
            "Lag phase cutoff",
            0.01,
            0.5,
            float(params0.get("lag_cutoff", 0.1)),
            0.01,
            format="%.2f",
            help="Fraction of maximum growth rate used to define the end of lag phase",
        )
    with exp_col:
        exp_cutoff = st.number_input(
            "Exponential phase cutoff",
            0.01,
            0.5,
            float(params0.get("exp_cutoff", 0.1)),
            0.01,
            format="%.2f",
            help="Fraction of maximum growth rate used to define the end of exponential phase",
        )

    st.write("")
    st.markdown("**'No Growth' Thresholds**")
    st.caption("Wells failing these criteria will be marked as no growth")

    col1, col2, col3 = st.columns(3)
    min_data_points = col1.number_input(
        "Minimum data points",
        1,
        100,
        int(params0.get("min_data_points", 5)),
        1,
        help="Minimum number of valid data points required for growth analysis",
    )
    min_signal_to_noise = col2.number_input(
        "Minimum signal-to-noise ratio",
        0.1,
        100.0,
        float(params0.get("min_signal_to_noise", 5.0)),
        0.1,
        help="Minimum ratio of maximum to minimum OD600 signal (filters out flat curves)",
    )
    min_growth_rate = col3.number_input(
        "Minimum growth rate (1/h)",
        0.0,
        1.0,
        float(params0.get("min_growth_rate", 0.001)),
        0.0001,
        format="%.4f",
        help="Minimum specific growth rate to be considered growth (wells with lower rates are marked as no growth)",
    )

# Build final params dict
params = dict(
    time_unit=str(time_unit),
    pathlength_cm_=float(pl_cm),
    clip_time_series=clip_time_series,
    remove_wells=remove_wells,
    blank=bool(blank),
    window_points=int(window_points),
    lag_cutoff=float(lag_cutoff),
    exp_cutoff=float(exp_cutoff),
    sg_window=int(params0.get("sg_window", 15)),
    sg_poly=int(params0.get("sg_poly", 2)),
    min_data_points=int(min_data_points),
    min_signal_to_noise=float(min_signal_to_noise),
    min_growth_rate=float(min_growth_rate),
    growth_method=str(growth_method),
    model_type=str(model_type),
)

# Step 5: Click analyse
with st.container(border=True):
    st.header("Step 5. Click analyse")

    if st.button(
        "Update parameters and analyse selected plate",
        type="primary",
        width="stretch",
        disabled=not plate_id,
    ):
        rec = ss.plates.get(plate_id, {})
        if not rec.get("uploads"):
            st.error("No uploads found for this plate.")
        else:
            rec["params"] = params
            ss.plates[plate_id] = analyse_plate(rec)
            st.toast(f"Analysed {plate_id}", duration="infinite")
