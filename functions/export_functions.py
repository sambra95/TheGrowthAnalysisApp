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


# ---------------- ZIP builder ----------------
def build_export_zip(
    plates: dict,
    *,
    include_tables: bool,
    include_plate_view: bool,
    include_baseline_plots: bool,
    include_well_plots: bool,
    well_graphs: list[str] | None = None,  # e.g. ["raw", "d1", "d2"]
    selected_plate_ids: list[str] | None = None,  # plates to include for well plots
    wells_by_plate: dict[str, list[str]] | None = None,  # {plate_id: [well,...]}
    add_annotations: bool = True,
    line_hours: float = 4.0,
    scale: int = 2,
    plot_width: int = 1200,
    plot_height: int = 800,
) -> bytes:
    """Build a ZIP of CSVs and static PNG plots based on selected options."""
    well_graphs = well_graphs or []
    selected_plate_ids = selected_plate_ids or []
    wells_by_plate = wells_by_plate or {}

    def _png(fig) -> bytes:
        return fig.to_image(
            format="png", width=plot_width, height=plot_height, scale=scale
        )

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

        # ---- Baseline plots (GLOBAL) ----
        if include_baseline_plots:
            # baseline source here matches your earlier code; adjust if your structure differs
            baseline_fig = plot_baseline(plates["test_data"]["baseline"])
            zf.writestr("plots/baseline.png", _png(baseline_fig))

        # ---- Plate-view plots ----
        if include_plate_view:
            # replicates is global, so keep it outside plate folders
            rep_fig = plot_replicates_by_sample(plates)
            zf.writestr("plots/replicates_by_sample.png", _png(rep_fig))

            for pid, p in plates.items():
                fig = plot_window_plate(p, line_hours=line_hours)
                zf.writestr(f"plots/plates/{pid}/window_plate.png", _png(fig))

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

                    plate_dir = f"plots/plates/{pid}/wells"

                    if "raw" in well_graphs:
                        growth_stats = (p.get("growth_stats") or {}).get(well) or {}
                        lag_end = growth_stats.get("lag_phase_end")
                        exp_end = growth_stats.get("exponential_phase_end")

                        fig = go.Figure(plot_window_single(processed, well))

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
                        zf.writestr(f"{plate_dir}/growth_curves/{well}.png", _png(fig))

                    if "d1" in well_graphs:
                        fig = plot_window_single_d1(
                            p,
                            well,
                            sg_window=sg_w,
                            sg_poly=sg_p,
                            frac_peak=0.20,
                            add_fit=False,
                        )
                        zf.writestr(f"{plate_dir}/curves_d1/{well}.png", _png(fig))

                    if "d2" in well_graphs:
                        fig = plot_window_single_d2(
                            p, well, sg_window=sg_w, sg_poly=sg_p, add_fit=False
                        )
                        zf.writestr(
                            f"{plate_dir}/curves_d2/{well}.png",
                            _png(fig),
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

    with st.container(border=True):
        st.subheader("Export")

        with st.form("export_form"):
            # ---- Plot Options Section ----
            with st.container(border=True):
                st.markdown("**Plot Options**")

                c_plate = st.checkbox("Include plate-view plots", value=True)
                c_base = st.checkbox("Include baseline plots", value=True)
                c_well = st.checkbox("Include well level plots", value=False)
                c_add_annotations = st.checkbox(
                    "Add annotations to well plots", value=True
                )

                # Plot dimensions
                st.markdown("**Plot Dimensions**")
                col1, col2 = st.columns(2)
                plot_width = col1.number_input(
                    "Width (px)",
                    min_value=400,
                    max_value=3000,
                    value=1200,
                    step=100,
                    help="Width of exported plots in pixels",
                )
                plot_height = col2.number_input(
                    "Height (px)",
                    min_value=300,
                    max_value=2500,
                    value=800,
                    step=100,
                    help="Height of exported plots in pixels",
                )

                well_graphs = []
                selected_plate_ids = []
                wells_by_plate: dict[str, list[str]] = {}

                with st.expander("Well plot options", expanded=True):
                    well_graphs = st.multiselect(
                        "Which well graphs to include",
                        options=["raw", "d1", "d2"],
                        default=["raw", "d1", "d2"],
                        help="raw = annotated window plot; d1/d2 = derivative plots",
                    )

                    selected_plate_ids = st.multiselect(
                        "Which plates to include",
                        options=plate_ids,
                        default=plate_ids,
                    )

                    for pid in selected_plate_ids:
                        processed = (plates.get(pid) or {}).get("processed_data") or {}
                        available_wells = sorted(processed.keys())

                        st.markdown(f"**{pid}**")
                        all_wells = st.checkbox(
                            "Include all wells",
                            value=True,
                            key=f"all_wells__{pid}",
                        )
                        if all_wells:
                            wells_by_plate[pid] = []  # empty means "all"
                        else:
                            wells_by_plate[pid] = st.multiselect(
                                "Select wells",
                                options=available_wells,
                                default=available_wells[: min(8, len(available_wells))],
                                key=f"wells__{pid}",
                            )

            # ---- Table Options Section ----
            with st.container(border=True):
                st.markdown("**Table Options**")
                c_tables = st.checkbox("Include tabulated data", value=True)

            # Buttons in columns
            col_build, col_download = st.columns(2)

            with col_build:
                submitted = st.form_submit_button(
                    "Build ZIP", type="primary", use_container_width=True
                )

        if submitted:
            with st.spinner("Building ZIP file..."):
                st.session_state.export_zip_bytes = build_export_zip(
                    plates,
                    include_tables=c_tables,
                    include_plate_view=c_plate,
                    include_baseline_plots=c_base,
                    include_well_plots=c_well,
                    well_graphs=well_graphs,
                    selected_plate_ids=selected_plate_ids,
                    wells_by_plate=wells_by_plate,
                    add_annotations=c_add_annotations,
                    plot_width=plot_width,
                    plot_height=plot_height,
                )
            st.success("ZIP file ready for download!")

        # Download button - always visible but only enabled when ZIP is ready
        col1, col2 = st.columns(2)
        with col2:
            if st.session_state.export_zip_bytes is not None:
                st.download_button(
                    "Download export.zip",
                    data=st.session_state.export_zip_bytes,
                    file_name="export.zip",
                    mime="application/zip",
                    use_container_width=True,
                    type="primary",
                )
            else:
                st.button(
                    "Download export.zip",
                    disabled=True,
                    use_container_width=True,
                    type="secondary",
                    help="Click 'Build ZIP' first to generate the export file",
                )
