"""Export functionality for downloadable tables and plots."""

import io
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import growthcurves.plot as gc_plot

from functions.plotting_functions import (
    plot_baseline,
    plot_model_fit_single,
    plot_replicates_by_sample,
    plot_window_plate,
    plot_window_single,
    plot_window_single_d1,
    plot_window_single_d2,
    _finite_sorted_xy,
    is_bad_fit,
)


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    """Return plates from session state, or stop with a warning."""
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.info("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Export helpers ----------------
def _processed_wide_for_plate(p: dict, *, value_col: str) -> pd.DataFrame:
    """Return a wide, time-indexed DataFrame with one column per well."""
    frames = []
    for well, d in (p.get("processed_data") or {}).items():
        if d is None or d.empty:
            continue
        if "Time" not in d.columns or value_col not in d.columns:
            continue
        frames.append(d[["Time", value_col]].rename(columns={value_col: well}))

    if not frames:
        return pd.DataFrame()

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="Time", how="outer")

    return out.sort_values("Time").reset_index(drop=True)


def _growth_stats_per_well_df(p: dict) -> pd.DataFrame:
    """Return growth stats per well as a tidy DataFrame with sample names."""
    nm_by_well = p.get("name") or {}
    df = (
        pd.DataFrame.from_dict(p.get("growth_stats") or {}, orient="index")
        .rename_axis("well")
        .reset_index()
    )
    if not df.empty:
        # Add Sample Name column after well column
        df.insert(
            1,
            "Sample Name",
            df["well"].map(lambda w: (nm_by_well.get(w) or "").strip()),
        )
        # Remove internal columns not meant for export
        internal_cols = ["_lasso_update_time", "lasso_t_min", "lasso_t_max"]
        df = df.drop(
            columns=[c for c in internal_cols if c in df.columns], errors="ignore"
        )
    return df


def _growth_stats_mean_for_sample_df(p: dict) -> pd.DataFrame:
    """Return growth stats averaged per sample name with mean and std columns."""
    nm_by_well = p.get("name") or {}
    gs = _growth_stats_per_well_df(p)
    if gs.empty:
        return gs

    gs["Sample Name"] = gs["well"].map(lambda w: (nm_by_well.get(w) or "").strip())
    num = [c for c in gs.columns if pd.api.types.is_numeric_dtype(gs[c])]

    # Calculate both mean and std
    agg_funcs = {col: ["mean", "std"] for col in num}
    result = gs.groupby("Sample Name")[num].agg(["mean", "std"])

    # Flatten the multi-level columns: convert (col, 'mean') to 'col_mean'
    result.columns = ["_".join(col).strip() for col in result.columns.values]

    return result.reset_index()


def _analysis_params_df(p: dict) -> pd.DataFrame:
    """Return analysis parameters as a DataFrame."""
    params = p.get("params") or {}
    if not params:
        return pd.DataFrame()

    # Format parameters for display
    data = {
        "Parameter": [],
        "Value": [],
    }

    # Extract parameters with readable names
    param_mapping = {
        "read_interval_min": "Read interval (minutes)",
        "pathlength_cm_": "Pathlength (cm)",
        "clip_time_series": "Time series clip (hours)",
        "remove_wells": "Excluded wells",
        "blank": "Blank subtraction",
        "growth_method": "Analysis method",
        "model_type": "Growth model",
        "window_points": "Window size (points)",
        "sg_window": "Savitzky-Golay window",
        "sg_poly": "Savitzky-Golay polynomial order",
        "lag_cutoff": "Lag phase cutoff (fraction of μ_max)",
        "exp_cutoff": "Exponential phase cutoff (fraction of μ_max)",
        "min_data_points": "Minimum data points",
        "min_signal_to_noise": "Minimum signal-to-noise ratio",
    }

    for key, label in param_mapping.items():
        if key in params:
            value = params[key]
            # Format special cases
            if key == "clip_time_series" and isinstance(value, (tuple, list)):
                value = f"{value[0]} - {value[1]}"
            elif key == "remove_wells":
                if value is False or not value:
                    value = "None"
                else:
                    value = ", ".join(value)
            elif key == "blank":
                value = "Yes" if value else "No"

            data["Parameter"].append(label)
            data["Value"].append(str(value))

    return pd.DataFrame(data)


# ---------------- ZIP builder ----------------
def build_export_zip(
    plates: dict,
    *,
    include_baseline_corrected: bool,
    include_stats_per_well: bool,
    include_stats_per_sample: bool,
    include_params: bool,
    include_plate_view: bool,
    include_baseline_plots: bool,
    include_replicates: bool,
    include_well_plots: bool,
    well_graphs: list[str] | None = None,  # e.g. ["OD", "1st Derivative", "2nd Derivative"]
    selected_plate_ids: list[str] | None = None,  # plates to include for well plots
    wells_by_plate: dict[str, list[str]] | None = None,  # {plate_id: [well,...]}
    add_annotations: bool = True,
    scale: int = 2,
    baseline_width: int = 1200,
    baseline_height: int = 800,
    plate_width: int = 1200,
    plate_height: int = 800,
    well_width: int = 1200,
    well_height: int = 800,
) -> bytes:
    """Build a ZIP of CSVs and static PNG plots based on selected options."""
    well_graphs = well_graphs or []
    selected_plate_ids = selected_plate_ids or []
    wells_by_plate = wells_by_plate or {}

    def _png(fig, width: int, height: int) -> bytes:
        return fig.to_image(format="png", width=width, height=height, scale=scale)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        # ---- Tables (GLOBAL) ----
        for pid, p in plates.items():
            if include_baseline_corrected:
                wide = _processed_wide_for_plate(p, value_col="baseline_corrected")
                if not wide.empty:
                    zf.writestr(
                        f"tables/{pid}_processed_baseline_corrected.csv",
                        wide.to_csv(index=False),
                    )

            if include_stats_per_well:
                per_well_df = _growth_stats_per_well_df(p)
                if not per_well_df.empty:
                    zf.writestr(
                        f"tables/{pid}_growth_stats_per_well.csv",
                        per_well_df.to_csv(index=False),
                    )

            if include_stats_per_sample:
                mean_df = _growth_stats_mean_for_sample_df(p)
                if not mean_df.empty:
                    zf.writestr(
                        f"tables/{pid}_growth_stats_mean_for_sample.csv",
                        mean_df.to_csv(index=False),
                    )

        # ---- Analysis Parameters ----
        if include_params:
            for pid, p in plates.items():
                params_df = _analysis_params_df(p)
                if not params_df.empty:
                    zf.writestr(
                        f"tables/{pid}_analysis_parameters.csv",
                        params_df.to_csv(index=False),
                    )

        # ---- Baseline plots (GLOBAL) ----
        if include_baseline_plots:
            # baseline source here matches your earlier code; adjust if your structure differs
            test_data = plates.get("test_data", {})
            baseline = test_data.get("baseline")
            name_by_well = test_data.get("name", {})
            if baseline is not None and not baseline.empty:
                baseline_fig = plot_baseline(baseline, name_by_well=name_by_well)
                if baseline_fig is not None:
                    zf.writestr(
                        "plots/baseline.png",
                        _png(baseline_fig, baseline_width, baseline_height),
                    )

        # ---- Replicates plot (GLOBAL) ----
        if include_replicates:
            rep_fig = plot_replicates_by_sample(plates)
            if rep_fig is not None:
                zf.writestr(
                    "plots/replicates_by_sample.png",
                    _png(rep_fig, plate_width, plate_height),
                )

        # ---- Plate-view plots ----
        if include_plate_view:
            for pid, p in plates.items():
                fig = plot_window_plate(p)
                if fig is not None:
                    zf.writestr(
                        f"plots/{pid}/window_plate.png",
                        _png(fig, plate_width, plate_height),
                    )

        # ---- Well-level plots ----
        if include_well_plots and selected_plate_ids and well_graphs:
            for pid in selected_plate_ids:
                p = plates.get(pid)
                if not p:
                    continue

                processed = p.get("processed_data") or {}
                if not processed:
                    continue

                params = p.get("params") or {}
                sg_w = int(params.get("sg_window", 11))
                sg_p = int(params.get("sg_poly", 2))

                # use requested wells; default to all available if empty
                wells = wells_by_plate.get(pid) or list(processed.keys())

                for well in wells:
                    if well not in processed:
                        continue

                    plate_dir = f"plots/{pid}/wells"

                    if "OD" in well_graphs:
                        all_growth_stats = p.get("growth_stats") or {}
                        gs = all_growth_stats.get(well) or {}
                        fit_parameters = p.get("fit_parameters") or {}

                        # Get the processed data for this well
                        d = processed.get(well)

                        if d is not None and not d.empty:
                            # Get time and OD data
                            t_raw, y_raw = _finite_sorted_xy(
                                d["Time"].to_numpy(), d["baseline_corrected"].to_numpy()
                            )

                            if t_raw.size > 0:
                                # Convert time to display unit
                                t_display = t_raw

                                # Create base plot using growthcurves
                                fig = gc_plot.create_base_plot(t_display, y_raw, scale="linear")

                                # Annotate plot if requested
                                if add_annotations and not is_bad_fit(gs) and gs:
                                    # Get stats from dictionary
                                    exp_phase_start = gs.get("exp_phase_start")
                                    exp_phase_end = gs.get("exp_phase_end")
                                    time_at_umax = gs.get("time_at_umax")
                                    od_at_umax = gs.get("od_at_umax")
                                    max_od = gs.get("max_od")

                                    # Convert time values to display units
                                    if exp_phase_start is not None:
                                        exp_phase_start = exp_phase_start
                                    if exp_phase_end is not None:
                                        exp_phase_end = exp_phase_end
                                    if time_at_umax is not None:
                                        time_at_umax = time_at_umax

                                    # Prepare umax point
                                    umax_point = None
                                    if time_at_umax is not None and od_at_umax is not None:
                                        umax_point = (time_at_umax, od_at_umax)

                                    fit_result = fit_parameters.get(well)

                                    # Annotate plot
                                    fig = gc_plot.annotate_plot(
                                        fig,
                                        phase_boundaries=(exp_phase_start, exp_phase_end) if exp_phase_start and exp_phase_end else None,
                                        time_umax=time_at_umax,
                                        od_umax=od_at_umax,
                                        od_max=max_od,
                                        umax_point=umax_point,
                                        fitted_model=fit_result,
                                        scale="linear",
                                    )

                                # Update axis labels
                                time_label = "Time (hours)"
                                fig.update_xaxes(title=time_label, showgrid=False)
                                fig.update_yaxes(title="OD600 (baseline-corrected)", showgrid=False)

                                zf.writestr(
                                    f"{plate_dir}/growth_curves/{well}.png",
                                    _png(fig, well_width, well_height),
                                )

                    # Only export derivative plots for sliding window method
                    if "1st Derivative" in well_graphs or "2nd Derivative" in well_graphs:
                        gs = (p.get("growth_stats") or {}).get(well) or {}
                        fit_method = gs.get("fit_method", "sliding_window")
                        is_model_fit = fit_method and "model_fitting" in str(fit_method)

                        if not is_model_fit:
                            if "1st Derivative" in well_graphs:
                                fig = plot_window_single_d1(
                                    p,
                                    well,
                                    sg_window=sg_w,
                                    sg_poly=sg_p,
                                    frac_peak=0.20,
                                    add_fit=False,
                                )
                                if fig is not None:
                                    zf.writestr(
                                        f"{plate_dir}/curves_d1/{well}.png",
                                        _png(fig, well_width, well_height),
                                    )

                            if "2nd Derivative" in well_graphs:
                                fig = plot_window_single_d2(
                                    p, well, sg_window=sg_w, sg_poly=sg_p, add_fit=False
                                )
                                if fig is not None:
                                    zf.writestr(
                                        f"{plate_dir}/curves_d2/{well}.png",
                                        _png(fig, well_width, well_height),
                                    )

    buf.seek(0)
    return buf.getvalue()


# ---------------- UI ----------------
@st.fragment
def ui_export(plates: dict):
    """Render export controls and a ZIP download button."""
    plate_ids = list(plates.keys())

    # Initialize session state for ZIP bytes
    if "export_zip_bytes" not in st.session_state:
        st.session_state.export_zip_bytes = None

    # ---- Plot Options in 2x2 Grid ----
    row1_col1, row1_col2 = st.columns(2)

    # Tables
    with row1_col1:
        with st.container(border=True):
            st.header("Tabulated Data")

            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_baseline_corrected = st.checkbox(
                    "Baseline-corrected", value=True, key="table_baseline_corrected"
                )
            with desc_col:
                st.caption("Time series OD600 values for each well")

            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_stats_per_well = st.checkbox(
                    "Stats per well", value=True, key="table_stats_per_well"
                )
            with desc_col:
                st.caption(
                    "Max growth rate, lag time, max OD, and phase boundaries per well"
                )

            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_stats_per_sample = st.checkbox(
                    "Stats per sample", value=True, key="table_stats_per_sample"
                )
            with desc_col:
                st.caption("Statistics averaged across replicates")

            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_params = st.checkbox(
                    "Analysis parameters", value=True, key="table_params"
                )
            with desc_col:
                st.caption(
                    "All analysis settings used (read interval, pathlength, etc.)"
                )

    # Well Level Plots
    with row1_col2:
        with st.container(border=True):
            st.header("Well Level Plots")
            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_well = st.checkbox(
                    "Include well plots", value=False, key="well_checkbox"
                )
            with desc_col:
                st.caption(
                    "Individual well growth curves with annotations and derivative plots"
                )

            if c_well:
                cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
                with cb_col:
                    c_add_annotations = st.checkbox(
                        "Annotations",
                        value=True,
                        key="annotations_checkbox",
                    )
                with desc_col:
                    st.caption("Draw phase boundary lines on well plots")

                col_w, col_h = st.columns(2)
                well_width = col_w.number_input(
                    "Width (px)",
                    min_value=400,
                    max_value=3000,
                    value=1200,
                    step=100,
                    key="well_width",
                )
                well_height = col_h.number_input(
                    "Height (px)",
                    min_value=300,
                    max_value=2500,
                    value=800,
                    step=100,
                    key="well_height",
                )

                well_graphs = st.multiselect(
                    "Well graph types",
                    options=["OD", "1st Derivative", "2nd Derivative"],
                    default=["OD", "1st Derivative", "2nd Derivative"],
                    key="well_graphs",
                )

                selected_plate_ids = st.multiselect(
                    "Plates to include",
                    options=plate_ids,
                    default=plate_ids,
                    key="selected_plates",
                )

                # Well selection within the same container
                wells_by_plate: dict[str, list[str]] = {}

                if selected_plate_ids:

                    include_all_wells = st.checkbox(
                        "Include all wells for all plates",
                        value=True,
                        key="include_all_wells_global",
                    )

                    if not include_all_wells:
                        for pid in selected_plate_ids:
                            processed = (plates.get(pid) or {}).get(
                                "processed_data"
                            ) or {}
                            available_wells = sorted(processed.keys())

                            wells_by_plate[pid] = st.multiselect(
                                f"Wells for {pid}",
                                options=available_wells,
                                default=available_wells[: min(3, len(available_wells))],
                                key=f"wells__{pid}",
                            )
                    else:
                        for pid in selected_plate_ids:
                            wells_by_plate[pid] = []  # empty means "all"
            else:
                # Set defaults when well plots are not included
                c_add_annotations = True
                well_width = 1200
                well_height = 800
                well_graphs = []
                selected_plate_ids = []
                wells_by_plate = {}

    # Global Plots
    with row1_col1:
        with st.container(border=True):
            st.header("Global Plots")

            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_base = st.checkbox("Baseline", value=True, key="baseline_checkbox")
            with desc_col:
                st.caption("Blank well OD measurements and mean baseline over time")

            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_plate = st.checkbox("Plate view", value=True, key="plate_checkbox")
            with desc_col:
                st.caption(
                    "96-well plate overview showing all wells with fitted growth curves"
                )

            cb_col, desc_col = st.columns([1, 3], vertical_alignment="center")
            with cb_col:
                c_replicates = st.checkbox(
                    "Replicates", value=True, key="replicates_checkbox"
                )
            with desc_col:
                st.caption("Replicate growth curves grouped by sample name")

            if c_base or c_plate or c_replicates:
                col_w, col_h = st.columns(2)
                global_width = col_w.number_input(
                    "Width (px)",
                    min_value=400,
                    max_value=3000,
                    value=1200,
                    step=100,
                    key="global_plot_width",
                )
                global_height = col_h.number_input(
                    "Height (px)",
                    min_value=300,
                    max_value=2500,
                    value=800,
                    step=100,
                    key="global_plot_height",
                )
            else:
                global_width = 1200
                global_height = 800

    # Lazy function that builds ZIP only when called by download button
    @st.cache_data(show_spinner="Building ZIP file...")
    def get_export_zip(
        _plates,
        include_baseline_corrected,
        include_stats_per_well,
        include_stats_per_sample,
        include_params,
        include_plate_view,
        include_baseline_plots,
        include_replicates,
        include_well_plots,
        well_graphs_tuple,
        selected_plates_tuple,
        wells_tuple,
        add_annotations,
        global_w,
        global_h,
        well_w,
        well_h,
    ):
        # Convert tuples back to appropriate types
        wells_dict = {}
        for k, v in wells_tuple:
            wells_dict[k] = list(v)

        return build_export_zip(
            _plates,
            include_baseline_corrected=include_baseline_corrected,
            include_stats_per_well=include_stats_per_well,
            include_stats_per_sample=include_stats_per_sample,
            include_params=include_params,
            include_plate_view=include_plate_view,
            include_baseline_plots=include_baseline_plots,
            include_replicates=include_replicates,
            include_well_plots=include_well_plots,
            well_graphs=list(well_graphs_tuple) if well_graphs_tuple else [],
            selected_plate_ids=(
                list(selected_plates_tuple) if selected_plates_tuple else []
            ),
            wells_by_plate=wells_dict,
            add_annotations=add_annotations,
            baseline_width=global_w,
            baseline_height=global_h,
            plate_width=global_w,
            plate_height=global_h,
            well_width=well_w,
            well_height=well_h,
        )

    # Convert lists/dicts to tuples for caching
    wells_tuple = tuple((k, tuple(v)) for k, v in sorted(wells_by_plate.items()))

    # Use data parameter with a callable to build ZIP only when download is clicked
    st.download_button(
        "Download Export ZIP",
        data=lambda: get_export_zip(
            plates,
            c_baseline_corrected,
            c_stats_per_well,
            c_stats_per_sample,
            c_params,
            c_plate,
            c_base,
            c_replicates,
            c_well,
            tuple(well_graphs) if well_graphs else (),
            tuple(selected_plate_ids) if selected_plate_ids else (),
            wells_tuple,
            c_add_annotations,
            global_width,
            global_height,
            well_width,
            well_height,
        ),
        file_name="export.zip",
        mime="application/zip",
        width="stretch",
        type="primary",
    )
