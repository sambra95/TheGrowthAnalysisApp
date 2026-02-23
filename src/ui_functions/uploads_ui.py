"""UI fragments for the Upload and Analyze page."""

import pandas as pd
import streamlit as st
from growthcurves.models import MODEL_REGISTRY

from src.functions.constants import COLS, DEFAULT_PARAMS, GRAY, ROWS
from src.functions.data_processing import analyse_plate, load_plate
from src.ui_functions.ui_components import (
    ui_method_visualization,
    ui_phase_boundary_visualization,
)
from src.functions.upload_functions import (
    build_symbol_grid,
    get_plate_preview_data,
    plate_params,
    validate_data_file,
    validate_plate_map_file,
)
from src.styling import growth_param_table_style, plate_table_style


def ui_upload_and_analyse_header():
    """Render page title and help popover."""
    title_col, popover_col = st.columns([9, 2])
    with title_col:
        st.title("Upload and Analyze")
    with popover_col:
        st.write("")
        with st.popover("Help", width="stretch"):
            st.markdown(
                """
**Workflow Overview — Upload & Analyse**

This is your starting point. Follow the 6 steps in order to upload your data and run the growth analysis.

**Step 1 — Upload data file**
Upload your plate reader Excel file. Click "Requirements" to see the expected format and download an example. Your file must have a **Time** column plus one column per well (e.g. A1, A2, ...).

**Step 2 — Upload plate map**
Upload a plate map Excel file that assigns sample names to each well. Wells with the same name are treated as replicates. Use **BLANK** for blank wells and leave cells empty to ignore wells. Click "Requirements" for the expected format and an example download.

**Step 3 — Match samples with names**
Click the button to load both files and match well IDs between the data and the plate map. The plate preview on the right will update to confirm which wells are included.

**Step 4 — Select preprocessing parameters**
Configure how the data is processed before analysis:
- **Time unit**: Set to match the unit in your data file (seconds, minutes, or hours)
- **Pathlength**: Your plate reader's optical path length, used to normalise OD to 1 cm
- **Blank subtraction**: Subtract the mean BLANK well OD from all other wells
- **Time range**: Restrict analysis to a specific window of the experiment
- **Exclude wells**: Manually remove specific wells (e.g. contaminated or failed wells)

The plate preview updates live — 🟩 included · 🟦 blank · 🟥 excluded by user · 🟧 missing from data file · ⬜ not in plate map.

**Step 5 — Select analysis parameters**
Choose how growth descriptors are calculated:
- **Model family & method**: Select a parametric model (mechanistic or phenomenological) or a non-parametric approach (Sliding Window or Spline). The visualisation below the selector shows the shape of the selected model.
- **Quality control filters**: Wells not meeting these thresholds are automatically marked as "no growth"
- **Phase boundary method**: Controls how the lag phase end and exponential phase end are determined

The table at the bottom shows exactly how each growth parameter will be calculated for your selected settings.

**Step 6 — Analyse**
Click the button to run the analysis. Once complete, navigate to the other pages using the top navigation bar to review and download your results.
"""
            )

    st.divider()


def render_plate_table(grid: pd.DataFrame):
    """Render an HTML table showing the plate status grid."""
    # Apply styling (moved to styling.py)
    plate_table_style()

    header = "".join(f"<th>{c}</th>" for c in grid.columns)
    rows_html = []
    for r in grid.index:
        cells = "".join(f"<td>{grid.loc[r, c]}</td>" for c in grid.columns)
        rows_html.append(f"<tr><th class='row'>{r}</th>{cells}</tr>")

    html = f"""
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


def ui_upload_files(ss):
    """Fragment for file upload controls."""
    u1, u2 = st.columns(2)
    with u1:
        with st.container(border=True):
            # Header row with requirements in top right
            header_col, req_col = st.columns([3, 1])
            with header_col:
                st.header("Step 1. Upload data file")
            with req_col:
                with st.popover("Requirements", width="stretch"):
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

                    st.markdown("")
                    with open("example_data/example_data.xlsx", "rb") as f:
                        st.download_button(
                            "Download example data file",
                            data=f.read(),
                            file_name="example_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width="stretch",
                            type="primary",
                            key="download_example_data",
                        )

            data_file = st.file_uploader(
                "Plate reader Excel (.xlsx/.xls)", ["xlsx", "xls"], key="data_up"
            )
    with u2:
        with st.container(border=True):
            # Header row with requirements in top right
            header_col, req_col = st.columns([3, 1])
            with header_col:
                st.header("Step 2. Upload plate map")
            with req_col:
                with st.popover("Requirements", width="stretch"):
                    st.markdown("**Format:**")
                    st.markdown("- Excel file (.xlsx or .xls)")
                    st.markdown("- 96-well plate layout")

                    st.markdown("**Required structure:**")
                    st.markdown("- **'rows'** column with labels A-H")
                    st.markdown("- Columns **1-12** for well positions")

                    st.markdown("**Well labels:**")
                    st.markdown("- Samples with the same name = replicates")
                    st.markdown("- Use **'BLANK'** for blank wells")
                    st.markdown("- Empty cells = wells to ignore")
                    st.markdown("- First **'_'** splits strain and condition labels")

                    st.divider()
                    st.markdown("**Example format:**")

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

                    st.markdown("")
                    with open("example_data/example_plate_map.xls", "rb") as f:
                        st.download_button(
                            "Download example plate map",
                            data=f.read(),
                            file_name="example_plate_map.xls",
                            mime="application/vnd.ms-excel",
                            width="stretch",
                            type="primary",
                            key="download_example_plate_map",
                        )

            map_file = st.file_uploader(
                "Plate map (.xls/.xlsx) with 'rows' column",
                ["xlsx", "xls"],
                key="map_up",
            )

    with st.container(border=True):
        st.header("Step 3. Match samples with names")
        if st.button(
            "Match samples with names",
            type="primary",
            width="stretch",
            disabled=not (data_file and map_file),
        ):
            # Validate data file
            is_valid_data, data_error = validate_data_file(data_file.getvalue())
            if not is_valid_data:
                st.toast(f"❌ Data file validation failed: {data_error}", icon="🚫")
                st.stop()

            # Validate plate map file
            is_valid_map, map_error = validate_plate_map_file(map_file.getvalue())
            if not is_valid_map:
                st.toast(f"❌ Plate map validation failed: {map_error}", icon="🚫")
                st.stop()

            # If both validations pass, load the plate
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
            st.toast(f"✅ Successfully loaded {plate_id}")


@st.fragment
def ui_preprocessing_params(ss):
    """Fragment for preprocessing parameters and plate preview."""
    ready = sorted(ss.plates)

    with st.container(border=True):
        st.header("Step 4. Select plate and preprocessing parameters")

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
                "Blank subtraction",
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
            # Preview grid in Step 4
            if plate_id:
                rec = ss.plates.get(plate_id, {})
                if rec.get("uploads"):
                    # Use lightweight preview function instead of full analysis
                    uploads = rec["uploads"]
                    plate_map, present = get_plate_preview_data(
                        plate_bytes=uploads["plate_bytes"],
                        data_bytes=uploads["data_bytes"],
                    )

                    grid = build_symbol_grid(
                        plate_map=plate_map,
                        present=present,
                        remove_wells=remove_wells,
                        blank=blank,
                    )

                    st.subheader(plate_id)
                    st.caption(
                        "**Included:** 🟩 sample · 🟦 blank  |  "
                        "**Excluded:** 🟥 removed by user · 🟧 not in data file · ⬜ not in plate map"
                    )
                    render_plate_table(grid)

            else:
                # Show blank plate (all gray) when no files uploaded
                blank_grid = pd.DataFrame(GRAY, index=ROWS, columns=COLS)
                st.subheader("Plate Preview")
                st.caption(
                    "**Included:** 🟩 sample · 🟦 blank  |  "
                    "**Excluded:** 🟥 removed by user · 🟧 not in data file · ⬜ not in plate map"
                )
                render_plate_table(blank_grid)

    # Store selected values in session state for access by other fragments
    if plate_id:
        ss.setdefault("step3_params", {})
        ss["step3_params"]["plate_id"] = plate_id
        ss["step3_params"]["time_unit"] = time_unit
        ss["step3_params"]["pl_cm"] = pl_cm
        ss["step3_params"]["blank"] = blank
        ss["step3_params"]["clip_time_series"] = clip_time_series
        ss["step3_params"]["remove_wells"] = remove_wells
        ss["step3_params"]["params0"] = params0


# Helper functions for analysis parameters
def _get_model_display_name(model_code: str) -> str:
    """Convert model code to display name."""
    display_names = {
        "mech_logistic": "Logistic",
        "mech_gompertz": "Gompertz",
        "mech_richards": "Richards",
        "mech_baranyi": "Baranyi-Roberts",
        "phenom_logistic": "Logistic",
        "phenom_gompertz": "Gompertz",
        "phenom_gompertz_modified": "Modified Gompertz",
        "phenom_richards": "Richards",
        "sliding_window": "Sliding Window",
        "spline": "Spline",
    }
    return display_names.get(model_code, model_code)


def _ui_model_selection(params0: dict):
    """Render model family and growth method selection UI."""
    stored_method = params0.get("growth_method", "Sliding Window")
    stored_model_family = params0.get("model_family", "mechanistic")
    stored_model_type = params0.get("model_type", "mech_logistic")

    st.caption("Select the model family and growth descriptor method:")

    family_col, method_col, param_col = st.columns(3)

    # Determine default family index from stored state
    if stored_method in ["Sliding Window", "Spline"]:
        default_family_idx = 2
    elif stored_model_family == "mechanistic":
        default_family_idx = 0
    else:
        default_family_idx = 1

    with family_col:
        model_family = st.selectbox(
            "Model family",
            options=[
                "Mechanistic parametric",
                "Phenomenological parametric",
                "Non-parametric",
            ],
            index=default_family_idx,
            help="Mechanistic models use ODE-based biological principles. Phenomenological models describe growth patterns empirically. Non-parametric methods are data-driven without a fixed curve shape.",
        )

    if model_family == "Mechanistic parametric":
        model_family_internal = "mechanistic"
    elif model_family == "Phenomenological parametric":
        model_family_internal = "phenomenological"
    else:
        model_family_internal = "non_parametric"

    # Build method options from MODEL_REGISTRY
    method_options = []

    if model_family == "Mechanistic parametric":
        for model_code in MODEL_REGISTRY["mechanistic"]:
            method_options.append(
                (_get_model_display_name(model_code), model_code, "Model Fitting")
            )
    elif model_family == "Phenomenological parametric":
        for model_code in MODEL_REGISTRY["phenomenological"]:
            method_options.append(
                (_get_model_display_name(model_code), model_code, "Model Fitting")
            )
    else:  # Non-parametric
        for model_code in MODEL_REGISTRY["non_parametric"]:
            growth_method = (
                "Sliding Window" if model_code == "sliding_window" else "Spline"
            )
            method_options.append(
                (_get_model_display_name(model_code), model_code, growth_method)
            )

    # Determine default index
    default_idx = 0
    for i, (label, code, method) in enumerate(method_options):
        if stored_method in ["Sliding Window", "Spline"]:
            if code == "sliding_window" and stored_method == "Sliding Window":
                default_idx = i
                break
            elif code == "spline" and stored_method == "Spline":
                default_idx = i
                break
        elif stored_method == "Model Fitting":
            if code == stored_model_type:
                default_idx = i
                break

    with method_col:
        selected_method_label = st.selectbox(
            "Growth descriptor method",
            options=[m[0] for m in method_options],
            index=default_idx,
            help="Choose between non-parametric (data-driven) or parametric (model-based) approaches.",
        )

    # Extract internal codes
    growth_method = None
    model_type = None
    for label, code, method in method_options:
        if label == selected_method_label:
            growth_method = method
            if method == "Model Fitting":
                model_type = code
            break

    return model_family_internal, growth_method, model_type, param_col


def ui_model_params(growth_method: str, params0: dict, step4_prev: dict, param_col):
    """Render method-specific parameters (window size or spline smoothing)."""
    default_spline_s = step4_prev.get("spline_s", params0.get("spline_s", 1.0))
    if default_spline_s is None:
        default_spline_s = 1.0

    with param_col:
        if growth_method == "Sliding Window":
            window_points = st.number_input(
                "Window size (points)",
                5,
                200,
                int(params0["window_points"]),
                1,
                help="Number of consecutive data points used for sliding window linear fit to determine maximum growth rate",
            )
            spline_s = float(default_spline_s)
        elif growth_method == "Spline":
            window_points = int(params0["window_points"])
            spline_s = st.number_input(
                "Spline smoothing factor (s)",
                0.001,
                None,
                float(default_spline_s),
                0.001,
                help="Lower values follow noise more closely; higher values produce smoother fits.",
            )
        else:  # Model Fitting
            window_points = int(params0["window_points"])
            spline_s = float(default_spline_s)

    return window_points, spline_s


def ui_qc_filters(params0: dict):
    """Render quality control filter inputs."""
    st.caption("Wells failing these criteria will be marked as no growth")

    col1, col2, col3, col4 = st.columns(4)
    min_data_points = col1.number_input(
        "Minimum data points",
        1,
        100,
        int(params0.get("min_data_points", 5)),
        1,
        help="Minimum number of valid data points required for growth analysis",
    )
    min_signal_to_noise = col2.number_input(
        "Minimum signal:noise",
        0.1,
        100.0,
        float(params0.get("min_signal_to_noise", 1.0)),
        0.1,
        help="Minimum ratio of maximum to minimum OD600 signal (filters out flat curves)",
    )
    min_od_increase = col3.number_input(
        "Minimum OD increase",
        0.0,
        None,
        float(params0.get("min_od_increase", 0.05)),
        0.001,
        format="%.3f",
        help="Minimum absolute increase in OD600 from baseline to be considered growth",
    )
    min_growth_rate = col4.number_input(
        "Minimum growth rate",
        0.0,
        None,
        float(params0.get("min_growth_rate", 0.001)),
        0.0001,
        format="%.4f",
        help="Minimum specific growth rate to be considered growth (wells with lower rates are marked as no growth)",
    )

    return min_data_points, min_signal_to_noise, min_od_increase, min_growth_rate


def ui_phase_boundaries(params0: dict):
    """Render phase boundary method selection UI."""
    st.caption(
        "Phase boundaries define when the lag phase ends and when the exponential phase ends."
    )
    phase_boundary_method = st.selectbox(
        "Phase boundary calculation",
        options=["threshold", "tangent"],
        index=(
            0 if params0.get("phase_boundary_method", "tangent") == "threshold" else 1
        ),
        format_func=lambda v: v.capitalize(),
        help="Threshold uses fractions of μ_max; tangent uses the tangent at μ_max to estimate exponential phase bounds.",
    )
    lag_cutoff = st.number_input(
        "Lag phase cutoff",
        0.01,
        0.5,
        float(params0.get("lag_cutoff", 0.5)),
        0.01,
        format="%.2f",
        disabled=phase_boundary_method == "tangent",
        help="Fraction of maximum growth rate used to define lag phase end (threshold mode).",
    )
    exp_cutoff = st.number_input(
        "Exponential phase cutoff",
        0.01,
        0.5,
        float(params0.get("exp_cutoff", 0.5)),
        0.01,
        format="%.2f",
        disabled=phase_boundary_method == "tangent",
        help="Fraction of maximum growth rate used to define exponential phase end (threshold mode).",
    )

    return phase_boundary_method, lag_cutoff, exp_cutoff


def ui_calculation_table(
    growth_method: str,
    model_type: str,
    model_family: str,
    phase_boundary_method: str,
    lag_cutoff: float,
    exp_cutoff: float,
    window_points: int,
):
    """Render the growth parameter calculations table."""
    if growth_method == "Model Fitting":
        mu_max_calc = "μ<sub>max</sub>"
        model_rmse_calc = "RMSE over entire curve"
        max_od_calc = "Maximum OD from fitted model"
    else:
        max_od_calc = "Maximum raw OD"
        if growth_method == "Sliding Window":
            mu_max_calc = "b"
            model_rmse_calc = f"RMSE over {window_points} point sliding-window"
        else:
            mu_max_calc = "Max spline derivative"
            model_rmse_calc = "RMSE over spline fit window (log phase)"

    if growth_method == "Model Fitting" and model_family == "mechanistic":
        intrinsic_calc = "Fitted intrinsic μ"
    else:
        intrinsic_calc = "N.a."

    if phase_boundary_method == "threshold":
        boundary_calc = f"Time at instantaneous μ > {lag_cutoff:.0%} μ<sub>max</sub>"
        exp_phase_end_calc = (
            f"Time at instantaneous μ < {exp_cutoff:.0%} μ<sub>max</sub>"
        )
    else:
        boundary_calc = "μ<sub>max</sub> tangent intersect with OD baseline"
        exp_phase_end_calc = "μ<sub>max</sub> tangent intersec with OD(max)"

    if growth_method == "Model Fitting" and model_type in {
        "phenom_logistic",
        "phenom_gompertz",
        "phenom_gompertz_modified",
        "phenom_richards",
    }:
        lag_time_calc = "λ"
    else:
        lag_time_calc = boundary_calc

    st.caption("Your selected settings will calculate growth parameters as follows:")
    # Apply styling (moved to styling.py)
    growth_param_table_style()
    st.markdown(
        f"""
<div class="growth-param-table">

| OD(max) | μ<sub>max</sub> | Intrinsic Growth Rate | Doubling Time | Lag Time | μ<sub>max</sub> Time | μ<sub>max</sub> OD | Exponential End Time | RMSE |
|---|---|---|---|---|---|---|---|---|
| {max_od_calc} | {mu_max_calc} | {intrinsic_calc} | ln(2) / μ<sub>max</sub> | {lag_time_calc} | Time at μ<sub>max</sub> | OD at μ<sub>max</sub> | {exp_phase_end_calc} | {model_rmse_calc} |

</div>
""",
        unsafe_allow_html=True,
    )


@st.fragment
def ui_analysis_params(ss):
    """Fragment for analysis parameters."""
    step3_params = ss.get("step3_params", {})
    step4_prev = ss.get("step4_params", {})
    params0 = step3_params.get("params0", DEFAULT_PARAMS)

    with st.container(border=True):
        st.header("Step 5. Select the analysis parameters")

        # Two columns: Model options | Phase boundary options
        model_col, boundary_col = st.columns(2, gap="large")

        with model_col:
            # Model selection
            model_family, growth_method, model_type, param_col = _ui_model_selection(
                params0
            )

            # Method-specific parameters
            window_points, spline_s = ui_model_params(
                growth_method, params0, step4_prev, param_col
            )

            st.write("")
            st.write("")

            # Quality control filters
            min_data_points, min_signal_to_noise, min_od_increase, min_growth_rate = (
                ui_qc_filters(params0)
            )

            st.write("")

        with boundary_col:
            # Phase boundary selection
            phase_boundary_method, lag_cutoff, exp_cutoff = ui_phase_boundaries(params0)
            st.write("")

        st.write("")

        # Visualization columns
        help_model_col, help_boundary_col = st.columns(2, gap="large")

        with help_model_col:
            model_fig = ui_method_visualization(growth_method, model_type)

        with help_boundary_col:
            boundary_image = ui_phase_boundary_visualization(phase_boundary_method)

        # Render the visualizations
        graph_col_model, graph_col_boundary = st.columns(2, gap="large")

        with graph_col_model:
            if isinstance(model_fig, str):
                st.image(model_fig, width="stretch")
            elif model_fig is not None:
                st.plotly_chart(model_fig, width="stretch", config={"staticPlot": True})

        with graph_col_boundary:
            if boundary_image is not None:
                st.image(boundary_image, width="stretch")

    # Store analysis parameters in session state
    ss.setdefault("step4_params", {})
    ss["step4_params"]["window_points"] = window_points
    ss["step4_params"]["lag_cutoff"] = lag_cutoff
    ss["step4_params"]["exp_cutoff"] = exp_cutoff
    ss["step4_params"]["min_data_points"] = min_data_points
    ss["step4_params"]["min_signal_to_noise"] = min_signal_to_noise
    ss["step4_params"]["min_od_increase"] = min_od_increase
    ss["step4_params"]["min_growth_rate"] = min_growth_rate
    ss["step4_params"]["growth_method"] = growth_method
    ss["step4_params"]["model_family"] = model_family
    ss["step4_params"]["model_type"] = model_type
    ss["step4_params"]["phase_boundary_method"] = phase_boundary_method
    ss["step4_params"]["spline_s"] = spline_s


def ui_analyse_button(ss):
    """Fragment for the analyze button."""
    # Get values from Step 4 and Step 5
    step3_params = ss.get("step3_params", {})
    step4_params = ss.get("step4_params", {})

    plate_id = step3_params.get("plate_id")
    time_unit = step3_params.get("time_unit", "hours")
    pl_cm = step3_params.get("pl_cm", 0.42)
    blank = step3_params.get("blank", True)
    clip_time_series = step3_params.get("clip_time_series", (0.0, 72.0))
    remove_wells = step3_params.get("remove_wells", False)
    params0 = step3_params.get("params0", DEFAULT_PARAMS)

    window_points = step4_params.get("window_points", 15)
    lag_cutoff = step4_params.get("lag_cutoff", 0.5)
    exp_cutoff = step4_params.get("exp_cutoff", 0.5)
    min_data_points = step4_params.get("min_data_points", 5)
    min_signal_to_noise = step4_params.get("min_signal_to_noise", 1.0)
    min_od_increase = step4_params.get("min_od_increase", 0.05)
    min_growth_rate = step4_params.get("min_growth_rate", 0.001)
    growth_method = step4_params.get("growth_method", "Sliding Window")
    model_family = step4_params.get("model_family", "mechanistic")
    model_type = step4_params.get("model_type", "mech_logistic")
    phase_boundary_method = step4_params.get("phase_boundary_method", "tangent")
    spline_s = step4_params.get("spline_s", None)

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
        min_od_increase=float(min_od_increase),
        min_growth_rate=float(min_growth_rate),
        growth_method=str(growth_method),
        model_family=str(model_family),
        model_type=str(model_type),
        phase_boundary_method=str(phase_boundary_method),
        spline_s=float(spline_s) if spline_s is not None else None,
    )

    with st.container(border=True):
        st.header("Step 6. Click analyse")

        ui_calculation_table(
            growth_method,
            model_type,
            model_family,
            phase_boundary_method,
            lag_cutoff,
            exp_cutoff,
            window_points,
        )

        st.write("")
        clicked = st.button(
            "Update parameters and analyse selected plate",
            type="primary",
            width="stretch",
            disabled=not plate_id,
        )
        if clicked:
            rec = ss.plates.get(plate_id, {})
            if not rec.get("uploads"):
                st.error("No uploads found for this plate.")
            else:
                rec["params"] = params
                with st.spinner("Analysing plate...", show_time=True):
                    ss.plates[plate_id] = analyse_plate(rec)
                st.toast(f"Analysed {plate_id}", duration="infinite")
