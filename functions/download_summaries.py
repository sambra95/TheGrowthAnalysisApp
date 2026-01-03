# functions/download_summaries.py

import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_sortables import sort_items

from functions.plotting_functions import (
    plot_growth_stats,
    plot_mean_growth,
    plot_replicates_scatter,
)


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Helpers ----------------
def _processed_df_for_sample(plates: dict, sample_name: str) -> pd.DataFrame:
    rows = []
    for pid, p in plates.items():
        nm_by_well = p.get("name") or {}
        for well, d in (p.get("processed_data") or {}).items():
            if (nm_by_well.get(well) or "").strip() == sample_name:
                rows.append(
                    d.assign(
                        plate=pid, well=well, key=f"{pid}_{well}", name=sample_name
                    )
                )

    if rows:
        return pd.concat(rows, ignore_index=True)

    return pd.DataFrame(
        columns=["Time", "baseline_corrected", "plate", "well", "key", "name"]
    )


def _build_growth_stats_long_df(
    plates: dict, sel_ids: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    sample_order: list[str] = []
    for sid in sel_ids:
        _, nm = sid.split("||", 1)
        if nm and nm not in sample_order:
            sample_order.append(nm)

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
    max_t = float(default)
    for p in plates.values():
        for d in (p.get("processed_data") or {}).values():
            if d is not None and not d.empty and "Time" in d.columns:
                max_t = max(max_t, float(d["Time"].max()))
    return max_t


def _unique_preserve_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------- UI: Growth summaries ----------------
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
    """
    Returns a long DF for curve-based plots (mean + replicates).

    Columns:
      - Sample Name
      - Time
      - baseline_corrected
      - plate, well, key (for hover)
    """
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


@st.fragment
def ui_growth_summaries(plates: dict):
    # -----------------------------
    # Build options DF on page load
    # -----------------------------
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
    with st.form("growth_combined_form"):
        # ---- your existing selection header + table (unchanged) ----
        st.subheader("Select Samples for Visualization")

        if has_split:
            h1, h2, h3, h4, h5, h6 = st.columns([1, 1, 1, 1, 1.2, 0.8])
            h1.markdown("**Plate**")
            h2.markdown("**Sample Name**")
            h3.markdown("**Strain**")
            h4.markdown("**Condition**")
            h5.markdown("**Wells**")
            h6.markdown("**Include**")
        else:
            h1, h2, h3, h4 = st.columns([1, 1, 1.2, 0.8])
            h1.markdown("**Plate**")
            h2.markdown("**Sample Name**")
            h3.markdown("**Wells**")
            h4.markdown("**Include**")

        with st.container(height=380):
            if has_split:
                for _, r in opt.iterrows():
                    c1, c2, c3, c4, c5, c6 = st.columns([1, 1, 1, 1, 1.2, 0.8])
                    c1.write(r["Plate"])
                    c2.write(r["Sample Name"])
                    c3.write(r["Strain"])
                    c4.write(r["Condition"])
                    c5.write(r["Wells"])
                    sel[r["_id"]] = c6.checkbox(
                        "",
                        value=sel[r["_id"]],
                        key=f"{sel_key}:{r['_id']}",
                    )
            else:
                for sid, plate, name, wells in zip(
                    ids, opt["Plate"], opt["Sample Name"], opt["Wells"]
                ):
                    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 0.8])
                    c1.write(plate)
                    c2.write(name)
                    c3.write(wells)
                    sel[sid] = c4.checkbox("", value=sel[sid], key=f"{sel_key}:{sid}")

        sel_ids = _selected_ids()
        sel_opt = _selected_opt_rows(sel_ids)
        sel_sample_names = (
            _unique_preserve_order(sel_opt["Sample Name"].astype(str).tolist())
            if not sel_opt.empty
            else []
        )

        col1, col2 = st.columns(2)
        # ============================================================
        # Box 1: Growth stats controls
        # ============================================================
        with col1.container(border=True):
            st.subheader("Plot Growth Statistics")

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
                use_container_width=True,
            )

        # ============================================================
        # Box 2: Mean + replicates controls
        # ============================================================
        with col2.container(border=True):
            st.subheader("Plot Mean and Replicate Growth Curves")

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
                use_container_width=True,
            )
            apply_reps = b2.form_submit_button(
                "Generate replicates plot",
                type="primary",
                use_container_width=True,
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
        st.plotly_chart(
            plot_growth_stats(
                long_df,
                x_col=x_col,
                legend_col=legend_col,
                x_order=x_ordered,
                legend_order=legend_ordered,
            ),
            use_container_width=True,
        )

    # Build the shared curves DF only if needed
    if apply_mean or apply_reps:
        curves_df = _build_growth_curves_long_df(plates, sel_sample_names)

    if apply_mean:
        st.plotly_chart(
            plot_mean_growth(
                curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1
            ),
            use_container_width=True,
        )

    if apply_reps:
        if curves_df.empty:
            st.info("No replicate data found.")
            return
        st.plotly_chart(
            plot_replicates_scatter(
                curves_df, curves_ordered, t_start=curves_t0, t_end=curves_t1
            ),
            use_container_width=True,
        )


# ---------------- Export helpers ----------------
def _processed_wide_for_plate(p: dict, *, value_col: str) -> pd.DataFrame:
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
    return (
        pd.DataFrame.from_dict(p.get("growth_stats") or {}, orient="index")
        .rename_axis("well")
        .reset_index()
    )


def _growth_stats_mean_for_sample_df(p: dict) -> pd.DataFrame:
    nm_by_well = p.get("name") or {}
    gs = _growth_stats_per_well_df(p)
    if gs.empty:
        return gs

    gs["Sample Name"] = gs["well"].map(lambda w: (nm_by_well.get(w) or "").strip())
    num = [c for c in gs.columns if pd.api.types.is_numeric_dtype(gs[c])]
    return gs.groupby("Sample Name")[num].mean().reset_index()


# ---------------- UI: Export ----------------
import io
import zipfile
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from functions.plotting_functions import (
    plot_baseline,
    plot_replicates_by_sample,
    plot_window_plate,
    plot_window_single,
    plot_window_single_d1,
    plot_window_single_d2,
    _vlines,
)


# ---------------- Export helpers ----------------
def _processed_wide_for_plate(p: dict, *, value_col: str) -> pd.DataFrame:
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
    return (
        pd.DataFrame.from_dict(p.get("growth_stats") or {}, orient="index")
        .rename_axis("well")
        .reset_index()
    )


def _growth_stats_mean_for_sample_df(p: dict) -> pd.DataFrame:
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
) -> bytes:
    well_graphs = well_graphs or []
    selected_plate_ids = selected_plate_ids or []
    wells_by_plate = wells_by_plate or {}

    def _png(fig) -> bytes:
        return fig.to_image(format="png", scale=scale)

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
    plate_ids = list(plates.keys())

    with st.container(border=True):
        st.subheader("Export")

        with st.form("export_form"):
            c_tables = st.checkbox("Include tabulated data", value=True)
            c_plate = st.checkbox("Include plate-view plots", value=True)
            c_base = st.checkbox("Include baseline plots", value=True)

            c_well = st.checkbox("Include well level plots", value=False)
            c_add_annotations = st.checkbox("Add annotations to well plots", value=True)

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

            submitted = st.form_submit_button(
                "Build ZIP", type="primary", use_container_width=True
            )

        if submitted:
            zip_bytes = build_export_zip(
                plates,
                include_tables=c_tables,
                include_plate_view=c_plate,
                include_baseline_plots=c_base,
                include_well_plots=c_well,
                well_graphs=well_graphs,
                selected_plate_ids=selected_plate_ids,
                wells_by_plate=wells_by_plate,
                add_annotations=c_add_annotations,
            )
            st.download_button(
                "Download export.zip",
                data=zip_bytes,
                file_name="export.zip",
                mime="application/zip",
                use_container_width=True,
            )
