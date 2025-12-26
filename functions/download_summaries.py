# functions/download_summaries.py

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_sortables import sort_items

from functions.plotting_functions import plot_growth_stats, plot_mean_growth


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
@st.fragment
def ui_growth_summaries(plates: dict):
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
    ids = opt["_id"].tolist()

    sel_key = "growth_combined_sel"
    sel = st.session_state.setdefault(sel_key, {})
    st.session_state[sel_key] = {sid: bool(sel.get(sid, False)) for sid in ids}
    sel = st.session_state[sel_key]

    max_t = _max_time_hours(plates)

    order_key = "growth_stats_sample_order"
    order_sig_key = "growth_stats_sample_order_sig"
    order_ver_key = "growth_stats_sample_order_ver"

    st.session_state.setdefault(order_key, [])
    st.session_state.setdefault(order_ver_key, 0)

    def _selected_ids():
        return [sid for sid in ids if sel.get(sid, False)]

    def _selected_samples(sel_ids):
        return _unique_preserve_order([sid.split("||", 1)[1] for sid in sel_ids])

    with st.form("growth_combined_form"):
        h1, h2, h3, h4 = st.columns([1, 1, 1.2, 0.8])
        h1.markdown("**Plate**")
        h2.markdown("**Sample Name**")
        h3.markdown("**Wells**")
        h4.markdown("**Include**")

        with st.container(height=380):
            for sid, plate, name, wells in zip(
                ids, opt["Plate"], opt["Sample Name"], opt["Wells"]
            ):
                c1, c2, c3, c4 = st.columns([1, 1, 1.2, 0.8])
                c1.write(plate)
                c2.write(name)
                c3.write(wells)
                sel[sid] = c4.checkbox("", value=sel[sid], key=f"{sel_key}:{sid}")

        t0, t1 = st.slider(
            "Plot time window (hours)",
            0.0,
            max_t,
            (0.0, min(72.0, max_t)),
            step=0.5,
        )

        sel_ids = _selected_ids()
        sel_samples = _selected_samples(sel_ids)

        cur_order = [s for s in st.session_state[order_key] if s in sel_samples]
        for s in sel_samples:
            if s not in cur_order:
                cur_order.append(s)
        st.session_state[order_key] = cur_order

        sig = tuple(sel_samples)
        if st.session_state.get(order_sig_key) != sig:
            st.session_state[order_sig_key] = sig
            st.session_state[order_ver_key] += 1

        if sel_samples:
            st.markdown("**Drag to set sample order (x-axis):**")
            st.session_state[order_key] = sort_items(
                st.session_state[order_key],
                key=f"growth_stats_sortable_{st.session_state[order_ver_key]}",
            )

        b1, b2, b3 = st.columns(3)
        apply_stats = b1.form_submit_button(
            "Generate growth stats plot",
            type="primary",
            use_container_width=True,
        )
        apply_mean = b2.form_submit_button(
            "Generate mean growth plot",
            type="primary",
            use_container_width=True,
        )
        apply_reps = b3.form_submit_button(
            "Generate replicates plot",
            type="primary",
            use_container_width=True,
        )

    ordered = [s for s in st.session_state[order_key] if s in sel_samples]

    if apply_stats:
        long_df, _ = _build_growth_stats_long_df(plates, sel_ids)
        st.plotly_chart(plot_growth_stats(long_df, ordered), use_container_width=True)

    if apply_mean:
        st.plotly_chart(
            plot_mean_growth(plates, ordered, t_start=t0, t_end=t1),
            use_container_width=True,
        )

    if apply_reps:
        frames = []
        for sample in sel_samples:
            d = _processed_df_for_sample(plates, sample)
            if not d.empty:
                frames.append(d.assign(sample=sample))

        if not frames:
            st.info("No replicate data found.")
            return

        reps_df = pd.concat(frames, ignore_index=True)
        reps_df = reps_df[(reps_df["Time"] >= t0) & (reps_df["Time"] <= t1)]

        fig = px.scatter(
            reps_df,
            x="Time",
            y="baseline_corrected",
            color="sample",
            hover_data=["plate", "well", "key"],
        )
        st.plotly_chart(fig, use_container_width=True)


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
