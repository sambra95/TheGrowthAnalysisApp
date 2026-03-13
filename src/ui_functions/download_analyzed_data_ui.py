import streamlit as st

from src.functions.export_functions import build_export_zip


def _render_tabulated_data_container():
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
            st.caption("All analysis settings used (read interval, pathlength, etc.)")

    return c_baseline_corrected, c_stats_per_well, c_stats_per_sample, c_params


def _render_well_level_plots_container(plates: dict, plate_ids: list[str]):
    with st.container(border=True):
        # Title and checkbox on the same line
        title_col, cb_col = st.columns([1, 2], vertical_alignment="center")
        with title_col:
            st.header("Well Level Plots")
        with cb_col:
            c_well = st.checkbox("Include well plots", value=False, key="well_checkbox")

        # Caption underneath the title
        st.caption(
            "Individual well growth curves with annotations and derivative plots"
        )

        if c_well:
            c_add_annotations = (
                True  # Always add annotations when well plots are enabled
            )

            with st.popover("Choose annotations to include", width="stretch"):
                st.caption("Choose which annotations to include on well plots:")

                # Create two columns: plot on left, checkboxes on right
                plot_col, checkbox_col = st.columns([2, 1])

                with plot_col:
                    # Show pre-generated demo plot image
                    st.image("info_plots/annotations.png", width="stretch")

                with checkbox_col:
                    annot_phase = st.checkbox(
                        "Phase boundaries",
                        value=True,
                        key="annot_phase_boundaries",
                    )
                    annot_umax_point = st.checkbox(
                        "Max growth rate point",
                        value=True,
                        key="annot_umax_point",
                    )
                    annot_od_max = st.checkbox(
                        "Max OD",
                        value=True,
                        key="annot_od_max",
                    )
                    annot_baseline_od = st.checkbox(
                        "Baseline OD",
                        value=True,
                        key="annot_baseline_od",
                    )
                    annot_tangent = st.checkbox(
                        "Tangent line at max growth",
                        value=False,
                        key="annot_tangent",
                    )
                    annot_fitted_model = st.checkbox(
                        "Fitted model curve",
                        value=True,
                        key="annot_fitted_model",
                    )

            # Single row for graph types, width, and height
            graph_col, width_col, height_col = st.columns([3, 1, 1])

            with graph_col:
                well_graphs = st.segmented_control(
                    "Well traces to include",
                    options=["Raw OD", "dOD/dt", "Specific Growth Rate"],
                    default=["Raw OD", "dOD/dt", "Specific Growth Rate"],
                    selection_mode="multi",
                    key="well_graphs",
                )

            with width_col:
                well_width = st.number_input(
                    "Width (px)",
                    min_value=400,
                    max_value=3000,
                    value=1200,
                    step=100,
                    key="well_width",
                )

            with height_col:
                well_height = st.number_input(
                    "Height (px)",
                    min_value=300,
                    max_value=2500,
                    value=800,
                    step=100,
                    key="well_height",
                )
            col1, col2 = st.columns((4, 1), vertical_alignment="center")
            selected_plate_ids = col1.multiselect(
                "Plates to include",
                options=plate_ids,
                default=plate_ids,
                key="selected_plates",
            )

            # Well selection within the same container
            wells_by_plate: dict[str, list[str]] = {}

            if selected_plate_ids:

                include_all_wells = col2.checkbox(
                    "Include all wells",
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
        else:
            # Set defaults when well plots are not included
            c_add_annotations = True
            annot_phase = True
            annot_umax_point = True
            annot_od_max = True
            annot_baseline_od = True
            annot_tangent = False
            annot_fitted_model = True
            well_width = 1200
            well_height = 800
            well_graphs = []
            selected_plate_ids = []
            wells_by_plate = {}

    return (
        c_well,
        c_add_annotations,
        annot_phase,
        annot_umax_point,
        annot_od_max,
        annot_baseline_od,
        annot_tangent,
        annot_fitted_model,
        well_width,
        well_height,
        well_graphs,
        selected_plate_ids,
        wells_by_plate,
    )


def _render_global_plots_container():
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

    return c_base, c_plate, c_replicates, global_width, global_height


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
    annot_phase,
    annot_umax_point,
    annot_od_max,
    annot_baseline_od,
    annot_tangent,
    annot_fitted_model,
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
        selected_plate_ids=list(selected_plates_tuple) if selected_plates_tuple else [],
        wells_by_plate=wells_dict,
        add_annotations=add_annotations,
        annot_phase=annot_phase,
        annot_umax_point=annot_umax_point,
        annot_od_max=annot_od_max,
        annot_baseline_od=annot_baseline_od,
        annot_tangent=annot_tangent,
        annot_fitted_model=annot_fitted_model,
        baseline_width=global_w,
        baseline_height=global_h,
        plate_width=global_w,
        plate_height=global_h,
        well_width=well_w,
        well_height=well_h,
    )
