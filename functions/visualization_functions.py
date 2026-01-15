"""Interactive visualizations for growth summaries."""

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

from functions.plotting_functions import (
    plot_growth_stats,
    plot_single_growth_stat,
    plot_mean_growth,
    plot_replicates_scatter,
)


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    """Return plates from session state, or stop with a warning."""
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Helpers ----------------
def _build_growth_stats_long_df(
    plates: dict, sel_ids: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Build a long DataFrame for growth stats plots and the sample order."""
    sample_order: list[str] = []
    for sid in sel_ids:
        _, nm = sid.split("||", 1)
        if nm and nm not in sample_order:
            sample_order.append(nm)

    metrics = [
        "Maximum U",
        "doubling_time",
        "Maximum OD600",
        "lag_phase_end",
        "exponential_phase_end",
        "t_mu",
        "y_mu",
        "t_peak",
    ]

    rows: list[dict] = []

    for sid in sel_ids:
        pid, nm = sid.split("||", 1)
        p = plates.get(pid) or {}
        name_map = p.get("name") or {}
        gs_map = p.get("growth_stats") or {}

        for well, well_nm in name_map.items():
            if (well_nm or "").strip() != nm:
                continue
            gs = gs_map.get(well) or {}
            if not gs:
                continue

            for m in metrics:
                rows.append(
                    {
                        "plate": pid,
                        "well": well,
                        "sample_name": nm,
                        "metric": m,
                        "value": float(gs.get(m, np.nan)),
                    }
                )

    return pd.DataFrame(rows), sample_order


def _max_time_hours(plates: dict, default: float = 72.0) -> float:
    """Return the maximum time (hours) across all processed data."""
    max_t = float(default)
    for p in plates.values():
        for d in (p.get("processed_data") or {}).values():
            if d is not None and not d.empty and "Time" in d.columns:
                max_t = max(max_t, float(d["Time"].max()))
    return max_t


def _unique_preserve_order(seq):
    """Return unique items in the original order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


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


def _build_growth_curves_long_df(plates: dict, sample_names: list[str]) -> pd.DataFrame:
    """Return a long DataFrame for mean/replicate curve plots."""
    if not sample_names:
        return pd.DataFrame(
            columns=[
                "Sample Name",
                "Time",
                "baseline_corrected",
                "plate",
                "well",
                "key",
            ]
        )

    sel = set(sample_names)
    rows = []

    for pid, _, well, nm, d, key in _iter_wells(plates):
        nm = (nm or "").strip()
        if nm in {"", "False", "BLANK"}:
            continue
        if nm not in sel:
            continue
        if d is None or d.empty:
            continue

        rows.append(
            pd.DataFrame(
                {
                    "Sample Name": nm,
                    "Time": d["Time"],
                    "baseline_corrected": d["baseline_corrected"],
                    "plate": pid,
                    "well": well,
                    "key": key,
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Sample Name",
                "Time",
                "baseline_corrected",
                "plate",
                "well",
                "key",
            ]
        )

    return pd.concat(rows, ignore_index=True)


# ---------------- UI: Growth summaries ----------------
@st.fragment
def ui_growth_summaries(plates: dict):
    """Render interactive growth summary plots with selection controls."""
    # Build options once per rerun.
    rows = []
    for pid, p in plates.items():
        by_name = {}
        for well, nm in (p.get("name") or {}).items():
            nm = (nm or "").strip()
            if not nm or nm in {"False", "BLANK"}:
                continue
            by_name.setdefault(nm, []).append(well)

        for nm, wells in by_name.items():
            rows.append((f"{pid}||{nm}", pid, nm, ", ".join(sorted(wells))))

    opt = (
        pd.DataFrame(rows, columns=["_id", "Plate", "Sample Name", "Wells"])
        .drop_duplicates("_id")
        .sort_values(["Plate", "Sample Name"], kind="stable")
        .reset_index(drop=True)
    )

    has_split = opt["Sample Name"].astype(str).str.contains("_", regex=False).any()
    if has_split and not opt.empty:
        sc = opt["Sample Name"].astype(str).str.split("_", n=1, expand=True)
        opt["Strain"] = sc[0]
        opt["Condition"] = sc[1].fillna("")

    ids = opt["_id"].tolist()

    # -----------------------------
    # Selection state
    # -----------------------------
    sel_key = "growth_combined_sel"
    sel = st.session_state.setdefault(sel_key, {})
    st.session_state[sel_key] = {sid: bool(sel.get(sid, False)) for sid in ids}
    sel = st.session_state[sel_key]

    max_t = _max_time_hours(plates)

    def _selected_ids():
        return [sid for sid in ids if sel.get(sid, False)]

    def _selected_opt_rows(sel_ids: list[str]) -> pd.DataFrame:
        if not sel_ids:
            return opt.iloc[0:0].copy()
        return opt[opt["_id"].isin(sel_ids)].copy()

    # -----------------------------
    # Order state (stats x-axis + legend)
    # -----------------------------
    x_order_key = "growth_stats_x_order"
    x_order_sig_key = "growth_stats_x_order_sig"
    x_order_ver_key = "growth_stats_x_order_ver"

    leg_order_key = "growth_stats_legend_order"
    leg_order_sig_key = "growth_stats_legend_order_sig"
    leg_order_ver_key = "growth_stats_legend_order_ver"

    st.session_state.setdefault(x_order_key, [])
    st.session_state.setdefault(x_order_ver_key, 0)
    st.session_state.setdefault(leg_order_key, [])
    st.session_state.setdefault(leg_order_ver_key, 0)

    # -----------------------------
    # Order state (curves sample order: mean+reps)
    # -----------------------------
    curves_order_key = "growth_curves_sample_order"
    curves_order_sig_key = "growth_curves_sample_order_sig"
    curves_order_ver_key = "growth_curves_sample_order_ver"
    st.session_state.setdefault(curves_order_key, [])
    st.session_state.setdefault(curves_order_ver_key, 0)

    # -----------------------------
    # UI
    # -----------------------------
    with st.form(
        "growth_combined_form",
        border=False,
    ):
        with st.container(border=True):
            st.header("Step 1. Select Samples for Visualization")

            # Add custom CSS to increase font size in data editor
            st.markdown("""
                <style>
                    div[data-testid="stDataFrame"] table {
                        font-size: 16px !important;
                    }
                    div[data-testid="stDataFrame"] thead th {
                        font-size: 17px !important;
                        font-weight: bold !important;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Prepare dataframe for display with selection column
            if has_split:
                display_cols = ["Plate", "Sample Name", "Strain", "Condition", "Wells"]
            else:
                display_cols = ["Plate", "Sample Name", "Wells"]

            # Add Select column with current selection state
            display_df = opt[display_cols].copy()
            display_df["Select"] = display_df.index.map(lambda i: sel.get(opt.iloc[i]["_id"], False))

            # Use data_editor for interactive table with visible cells
            edited_df = st.data_editor(
                display_df,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select samples for visualization",
                        default=False,
                    )
                },
                disabled=display_cols,  # Make all columns except Select read-only
                hide_index=True,
                width='stretch',
                height=400,
                key="sample_selection_table"
            )

            # Update selection state from edited dataframe
            for idx, row in edited_df.iterrows():
                sid = opt.iloc[idx]["_id"]
                sel[sid] = row["Select"]

            sel_ids = _selected_ids()
            sel_opt = _selected_opt_rows(sel_ids)
            sel_sample_names = (
                _unique_preserve_order(sel_opt["Sample Name"].astype(str).tolist())
                if not sel_opt.empty
                else []
            )

        col1, col_or, col2 = st.columns([1, 0.1, 1])
        # ============================================================
        # Box 1: Growth stats controls
        # ============================================================
        with col1.container(border=True):
            st.header("Step 2a. Plot Growth Statistics")

            x_choices = ["Sample Name"]
            group_choices = ["None"]
            if has_split:
                x_choices += ["Strain", "Condition"]
                group_choices += ["Strain", "Condition"]

            cA, cB = st.columns([1, 1])
            x_col = cA.selectbox(
                "X-axis column",
                options=x_choices,
                index=0,
                key="growth_stats_x_col",
            )
            legend_group = cB.selectbox(
                "Legend grouping",
                options=group_choices,
                index=0,
                key="growth_stats_legend_group",
            )
            legend_col = None if legend_group == "None" else legend_group

            x_vals = (
                _unique_preserve_order(sel_opt[x_col].astype(str).tolist())
                if (not sel_opt.empty and x_col in sel_opt.columns)
                else []
            )
            legend_vals = (
                _unique_preserve_order(sel_opt[legend_col].astype(str).tolist())
                if (legend_col and not sel_opt.empty and legend_col in sel_opt.columns)
                else []
            )

            # drag ordering: x-axis
            cur_x_order = [v for v in st.session_state[x_order_key] if v in x_vals]
            for v in x_vals:
                if v not in cur_x_order:
                    cur_x_order.append(v)
            st.session_state[x_order_key] = cur_x_order

            x_sig = (x_col, tuple(x_vals))
            if st.session_state.get(x_order_sig_key) != x_sig:
                st.session_state[x_order_sig_key] = x_sig
                st.session_state[x_order_ver_key] += 1

            if x_vals:
                st.markdown("**Drag to set x-axis order:**")
                st.session_state[x_order_key] = sort_items(
                    st.session_state[x_order_key],
                    key=f"growth_stats_x_sortable_{st.session_state[x_order_ver_key]}",
                )

            # drag ordering: legend
            if legend_col:
                cur_leg_order = [
                    v for v in st.session_state[leg_order_key] if v in legend_vals
                ]
                for v in legend_vals:
                    if v not in cur_leg_order:
                        cur_leg_order.append(v)
                st.session_state[leg_order_key] = cur_leg_order

                leg_sig = (legend_col, tuple(legend_vals))
                if st.session_state.get(leg_order_sig_key) != leg_sig:
                    st.session_state[leg_order_sig_key] = leg_sig
                    st.session_state[leg_order_ver_key] += 1

                if legend_vals:
                    st.markdown("**Drag to set legend order:**")
                    st.session_state[leg_order_key] = sort_items(
                        st.session_state[leg_order_key],
                        key=f"growth_stats_leg_sortable_{st.session_state[leg_order_ver_key]}",
                    )

            apply_stats = st.form_submit_button(
                "Generate growth stats plot",
                type="primary",
                width='stretch',
            )

        # ============================================================
        # Middle column: OR divider
        # ============================================================
        with col_or:
            st.markdown(
                "<h1 style='text-align: center; margin-top: 120px;'>OR</h1>",
                unsafe_allow_html=True,
            )

        # ============================================================
        # Box 2: Mean + replicates controls
        # ============================================================
        with col2.container(border=True):
            st.header("Step 2b. Plot Mean and Replicate Growth Curves")

            curves_t0, curves_t1 = st.slider(
                "Mean/replicates plot time window (hours)",
                0.0,
                max_t,
                (0.0, min(72.0, max_t)),
                step=0.5,
                key="growth_curves_time_window",
            )

            # drag ordering: sample names (specific to mean+reps)
            cur_curves_order = [
                v for v in st.session_state[curves_order_key] if v in sel_sample_names
            ]
            for v in sel_sample_names:
                if v not in cur_curves_order:
                    cur_curves_order.append(v)
            st.session_state[curves_order_key] = cur_curves_order

            curves_sig = tuple(sel_sample_names)
            if st.session_state.get(curves_order_sig_key) != curves_sig:
                st.session_state[curves_order_sig_key] = curves_sig
                st.session_state[curves_order_ver_key] += 1

            if sel_sample_names:
                st.markdown("**Drag to set Sample Name order (mean/replicates):**")
                st.session_state[curves_order_key] = sort_items(
                    st.session_state[curves_order_key],
                    key=f"growth_curves_sortable_{st.session_state[curves_order_ver_key]}",
                )

            b1, b2 = st.columns(2)
            apply_mean = b1.form_submit_button(
                "Generate mean growth plot",
                type="primary",
                width='stretch',
            )
            apply_reps = b2.form_submit_button(
                "Generate replicates plot",
                type="primary",
                width='stretch',
            )

    # -----------------------------
    # Resolve orders
    # -----------------------------
    x_ordered = [v for v in st.session_state[x_order_key] if v in x_vals]
    legend_ordered = (
        [v for v in st.session_state[leg_order_key] if v in legend_vals]
        if legend_col
        else []
    )
    curves_ordered = [
        v for v in st.session_state[curves_order_key] if v in sel_sample_names
    ]

    # -----------------------------
    # Plots
    # -----------------------------
    if apply_stats:
        long_df, _ = _build_growth_stats_long_df(plates, sel_ids)

        # Display each metric as a separate downloadable plot
        metrics = [
            "Maximum U",
            "doubling_time",
            "Maximum OD600",
            "lag_phase_end",
            "exponential_phase_end",
            "t_mu",
            "y_mu",
            "t_peak",
        ]

        for metric in metrics:
            metric_df = long_df[long_df["metric"] == metric].copy()
            if not metric_df.empty:
                fig = plot_single_growth_stat(
                    metric_df,
                    x_col=x_col,
                    legend_col=legend_col,
                    x_order=x_ordered,
                    legend_order=legend_ordered,
                )
                st.plotly_chart(fig, width='stretch', use_container_width=True)

    # Build the shared curves DF only if needed
    if apply_mean or apply_reps:
        curves_df = _build_growth_curves_long_df(plates, sel_sample_names)

    if apply_mean:
        st.plotly_chart(
            plot_mean_growth(
                curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1
            ),
            width='stretch',
        )

    if apply_reps:
        if curves_df.empty:
            st.info("No replicate data found.")
            return
        st.plotly_chart(
            plot_replicates_scatter(
                curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1
            ),
            width='stretch',
        )
