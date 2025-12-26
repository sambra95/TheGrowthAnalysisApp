# functions/download_summaries.py

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
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

    metrics = ["Maximum OD600", "Maximum U", "Lag Time (hours)"]
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
                # if your plot_growth_stats uses time window, pass stats_t0/stats_t1 too
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
            plot_replicates_scatter(curves_df, t_start=curves_t0, t_end=curves_t1),
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
@st.fragment
def ui_export(plates: dict):

    with st.container(border=True):

        st.subheader("Export Processed Data")

        value_col = st.selectbox(
            "Processed data value column",
            ["baseline_corrected", "raw", "od600", "value"],
        )

        gs_mode = st.radio(
            "Growth stats format",
            ["per_well", "mean_for_sample"],
            horizontal=True,
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for pid, p in plates.items():
                wide = _processed_wide_for_plate(p, value_col=value_col)
                if not wide.empty:
                    z.writestr(
                        f"{pid}/{pid}_processed_{value_col}_wide.csv",
                        wide.to_csv(index=False),
                    )

                if gs_mode == "per_well":
                    gs = _growth_stats_per_well_df(p)
                    if not gs.empty:
                        z.writestr(
                            f"{pid}/{pid}_growth_stats_per_well.csv",
                            gs.to_csv(index=False),
                        )
                else:
                    gs = _growth_stats_mean_for_sample_df(p)
                    if not gs.empty:
                        z.writestr(
                            f"{pid}/{pid}_growth_stats_mean_for_sample.csv",
                            gs.to_csv(index=False),
                        )

        buf.seek(0)

        st.download_button(
            "Download Tables",
            data=buf.getvalue(),
            file_name="growth_export_tables.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )
