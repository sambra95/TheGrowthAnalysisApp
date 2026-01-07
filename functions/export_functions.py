"""Export functionality for downloadable tables and plots."""

import io
import zipfile

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from functions.plotting_functions import (
    _vlines,
    plot_baseline,
    plot_replicates_by_sample,
    plot_window_plate,
    plot_window_single,
    plot_window_single_d1,
    plot_window_single_d2,
)


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    """Return plates from session state, or stop with a warning."""
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
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
    """Return growth stats per well as a tidy DataFrame."""
    return (
        pd.DataFrame.from_dict(p.get("growth_stats") or {}, orient="index")
        .rename_axis("well")
        .reset_index()
    )


def _growth_stats_mean_for_sample_df(p: dict) -> pd.DataFrame:
    """Return growth stats averaged per sample name."""
    nm_by_well = p.get("name") or {}
    gs = _growth_stats_per_well_df(p)
    if gs.empty:
        return gs

    gs["Sample Name"] = gs["well"].map(lambda w: (nm_by_well.get(w) or "").strip())
    num = [c for c in gs.columns if pd.api.types.is_numeric_dtype(gs[c])]
    return gs.groupby("Sample Name")[num].mean().reset_index()


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
        "window_points": "Window size (points)",
        "sg_window": "Savitzky-Golay window",
        "sg_poly": "Savitzky-Golay polynomial order",
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
    include_tables: bool,
    include_params: bool,
    include_plate_view: bool,
    include_baseline_plots: bool,
    include_well_plots: bool,
    well_graphs: list[str] | None = None,  # e.g. ["raw", "d1", "d2"]
    selected_plate_ids: list[str] | None = None,  # plates to include for well plots
    wells_by_plate: dict[str, list[str]] | None = None,  # {plate_id: [well,...]}
    add_annotations: bool = True,
    line_hours: float = 4.0,
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
        if include_tables:
            for pid, p in plates.items():
                wide = _processed_wide_for_plate(p, value_col="baseline_corrected")
                if not wide.empty:
                    zf.writestr(
                        f"tables/{pid}_processed_baseline_corrected.csv",
                        wide.to_csv(index=False),
                    )

                per_well_df = _growth_stats_per_well_df(p)
                if not per_well_df.empty:
                    zf.writestr(
                        f"tables/{pid}_growth_stats_per_well.csv",
                        per_well_df.to_csv(index=False),
                    )

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

        # ---- Plate-view plots ----
        if include_plate_view:
            # replicates is global, so keep it outside plate folders
            rep_fig = plot_replicates_by_sample(plates)
            if rep_fig is not None:
                zf.writestr(
                    "plots/replicates_by_sample.png",
                    _png(rep_fig, plate_width, plate_height),
                )

            for pid, p in plates.items():
                fig = plot_window_plate(p, line_hours=line_hours)
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

                sg_w = p.get("sg_window", 7)
                sg_p = p.get("sg_poly", 3)

                # use requested wells; default to all available if empty
                wells = wells_by_plate.get(pid) or list(processed.keys())

                for well in wells:
                    if well not in processed:
                        continue

                    plate_dir = f"plots/{pid}/wells"

                    if "raw" in well_graphs:
                        growth_stats = (p.get("growth_stats") or {}).get(well) or {}
                        lag_end = growth_stats.get("lag_phase_end")
                        exp_end = growth_stats.get("exponential_phase_end")

                        fig = plot_window_single(processed, well)
                        if fig is not None:
                            fig = go.Figure(fig)
                            if add_annotations:
                                _vlines(
                                    fig,
                                    processed,
                                    well,
                                    lag_end,
                                    exp_end,
                                    gs=growth_stats,
                                    line_hours=line_hours,
                                )
                            zf.writestr(
                                f"{plate_dir}/growth_curves/{well}.png",
                                _png(fig, well_width, well_height),
                            )

                    if "d1" in well_graphs:
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

                    if "d2" in well_graphs:
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

    # Tables Info
    with row1_col1:
        with st.container(border=True):
            st.header("Tables Included in Export")
            st.markdown(
                """
                The following tables are always included:
                - **Processed baseline-corrected data**: Time series OD600 values for each well
                - **Growth stats per well**: Maximum growth rate, lag time, max OD, and phase boundaries
                - **Growth stats mean per sample**: Averaged statistics across replicates
                - **Analysis parameters**: All analysis settings (read interval, pathlength, time clip, etc.)
                """
            )

    # Well Level Plots
    with row1_col2:
        with st.container(border=True):
            st.header("Well Level Plots")
            st.caption(
                "Individual well growth curves with annotations and derivative plots"
            )
            c_well = st.checkbox(
                "Include well level plots", value=False, key="well_checkbox"
            )
            c_add_annotations = st.checkbox(
                "Add annotations to well plots",
                value=True,
                key="annotations_checkbox",
            )

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
                options=["raw", "d1", "d2"],
                default=["raw", "d1", "d2"],
                help="raw = annotated window plot; d1/d2 = derivative plots",
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
                st.markdown("---")
                st.header("Well Selection")

                include_all_wells = st.checkbox(
                    "Include all wells for all plates",
                    value=True,
                    key="include_all_wells_global",
                )

                if not include_all_wells:
                    for pid in selected_plate_ids:
                        processed = (plates.get(pid) or {}).get("processed_data") or {}
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

    # Plate View Plots
    with row1_col1:
        with st.container(border=True):
            st.header("Plate View Plots")
            st.caption(
                "96-well plate overview showing all wells with fitted growth curves"
            )
            c_plate = st.checkbox(
                "Include plate-view plots", value=True, key="plate_checkbox"
            )

            col_w, col_h = st.columns(2)
            plate_width = col_w.number_input(
                "Width (px)",
                min_value=400,
                max_value=3000,
                value=1200,
                step=100,
                key="plate_width",
            )
            plate_height = col_h.number_input(
                "Height (px)",
                min_value=300,
                max_value=2500,
                value=800,
                step=100,
                key="plate_height",
            )

    # Baseline Plots
    with row1_col1:
        with st.container(border=True):
            st.header("Baseline Plots")
            st.caption("Blank well measurements and mean baseline over time")
            c_base = st.checkbox(
                "Include baseline plots", value=True, key="baseline_checkbox"
            )

            col_w, col_h = st.columns(2)
            baseline_width = col_w.number_input(
                "Width (px)",
                min_value=400,
                max_value=3000,
                value=1200,
                step=100,
                key="baseline_width",
            )
            baseline_height = col_h.number_input(
                "Height (px)",
                min_value=300,
                max_value=2500,
                value=800,
                step=100,
                key="baseline_height",
            )

    # Lazy function that builds ZIP only when called by download button
    @st.cache_data(show_spinner="Building ZIP file...")
    def get_export_zip(
        _plates,
        include_plate_view,
        include_baseline_plots,
        include_well_plots,
        well_graphs_tuple,
        selected_plates_tuple,
        wells_tuple,
        add_annotations,
        baseline_w,
        baseline_h,
        plate_w,
        plate_h,
        well_w,
        well_h,
    ):
        # Convert tuples back to appropriate types
        wells_dict = {}
        for k, v in wells_tuple:
            wells_dict[k] = list(v)

        return build_export_zip(
            _plates,
            include_tables=True,
            include_params=True,
            include_plate_view=include_plate_view,
            include_baseline_plots=include_baseline_plots,
            include_well_plots=include_well_plots,
            well_graphs=list(well_graphs_tuple) if well_graphs_tuple else [],
            selected_plate_ids=(
                list(selected_plates_tuple) if selected_plates_tuple else []
            ),
            wells_by_plate=wells_dict,
            add_annotations=add_annotations,
            baseline_width=baseline_w,
            baseline_height=baseline_h,
            plate_width=plate_w,
            plate_height=plate_h,
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
            c_plate,
            c_base,
            c_well,
            tuple(well_graphs) if well_graphs else (),
            tuple(selected_plate_ids) if selected_plate_ids else (),
            wells_tuple,
            c_add_annotations,
            baseline_width,
            baseline_height,
            plate_width,
            plate_height,
            well_width,
            well_height,
        ),
        file_name="export.zip",
        mime="application/zip",
        use_container_width=True,
        type="primary",
    )
