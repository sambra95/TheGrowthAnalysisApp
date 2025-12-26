# pages/analysis_shared.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_processing import ALL_WELLS
from plotting_functions import (
    _vlines,
    plot_growth_stats,
    plot_mean_growth,
    plot_replicates_by_sample,
    plot_window_plate,
    plot_window_single,
    plot_window_single_d1,
    plot_window_single_d2,
)

BAD_FIT = {
    "Maximum OD600": 0.0,
    "Maximum U": 0.0,
    "Lag Time (hours)": 0.0,
    "lag_phase_end": np.nan,
    "exponential_phase_end": np.nan,
    "t_mu": np.nan,
    "y_mu": np.nan,
    "b": np.nan,
    "t_peak": np.nan,
    "d1_fit": None,
}


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Selection + stats helpers ----------------
def _get_selected_points(event) -> tuple[np.ndarray, np.ndarray]:
    """Streamlit Plotly selection event -> arrays of selected x/y."""
    if event is None:
        return np.array([]), np.array([])

    sel = (
        event.get("selection")
        if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    if not sel:
        return np.array([]), np.array([])

    points = (
        sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    )
    if not points:
        return np.array([]), np.array([])

    xs = np.asarray([float(p["x"]) for p in points])
    ys = np.asarray([float(p["y"]) for p in points])
    return xs, ys


def update_growth_stats_from_lasso(
    plates: dict, pid: str, well: str, chart_key: str
) -> None:
    """
    Read the plotly selection payload from st.session_state[chart_key],
    fit y = m x + b on selected points,
    and write results into plates[pid]["growth_stats"][well].
    """
    xs, ys = _get_selected_points(st.session_state.get(chart_key))
    if xs.size < 2:
        return

    m, b = np.polyfit(xs, ys, deg=1)
    t_mu = float(xs.mean())
    y_mu = float(m * t_mu + b)

    gs = plates.setdefault(pid, {}).setdefault("growth_stats", {}).setdefault(well, {})
    gs["Maximum U"] = float(m)
    gs["t_mu"] = t_mu
    gs["y_mu"] = y_mu
    gs["b"] = float(b)


# ---------------- Data helpers ----------------
def _sample_names(plates: dict) -> list[str]:
    names: set[str] = set()
    for p in plates.values():
        for nm in (p.get("name") or {}).values():
            nm = (nm or "").strip()
            if nm and nm not in ("False", "BLANK"):
                names.add(nm)
    return sorted(names)


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


def _sg_params_for_plate(plates: dict, plate_id: str) -> tuple[int, int, int]:
    params = (plates.get(plate_id, {}) or {}).get("params") or {}
    return (
        int(params.get("sg_window", 11)),
        int(params.get("sg_poly", 2)),
        int(params.get("window_points", 15)),
    )


def _phase_controls(plate: dict, well: str, *, key: str):
    """Range slider (lag_end, exp_end) + 'No Growth' button. Writes into plate['growth_stats'][well]."""
    processed = (plate.get("processed_data") or {}).get(well)
    if processed is None or processed.empty:
        st.warning(f"No data for {well}")
        return np.nan, np.nan, True

    t = processed["Time"]
    t_min, t_max = float(t.min()), float(t.max())
    step = float(max((t_max - t_min) / 200.0, 0.01))

    growth_stats = (plate.get("growth_stats") or {}).setdefault(well, {})
    ss_key = f"phase__{key}"

    if ss_key not in st.session_state:
        lag0 = growth_stats.get("lag_phase_end")
        exp0 = growth_stats.get("exponential_phase_end")
        lag0 = float(lag0) if pd.notna(lag0) else t_min
        exp0 = (
            float(exp0) if pd.notna(exp0) else min(t_min + 0.5 * (t_max - t_min), t_max)
        )
        st.session_state[ss_key] = (lag0, exp0)

    c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
    with c1:
        lag_end, exp_end = st.slider(
            "Phase boundaries (hours): Lag end → Exponential end",
            t_min,
            t_max,
            st.session_state[ss_key],
            step=step,
            key=ss_key,
        )
    with c2:
        no_growth = st.button(
            "No Growth",
            use_container_width=True,
            type="primary",
            key=f"nogrowth__{key}",
        )

    growth_stats["lag_phase_end"] = float(lag_end)
    growth_stats["exponential_phase_end"] = float(exp_end)

    if no_growth:
        growth_stats.update(BAD_FIT.copy())
        st.rerun()
        return np.nan, np.nan, True

    return float(lag_end), float(exp_end), False


# ---------------- Window plot cache ----------------
@st.cache_data(show_spinner=False)
def _cached_window_single(processed_data: dict, well: str):
    return plot_window_single(processed_data, well)


# ---------------- Growth stats helpers ----------------
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


# ---------------- Fragments used by pages ----------------
@st.fragment
def ui_replicates(plates: dict):
    t_over, t_zoom = st.tabs(["Overview", "Per Sample View"])

    with t_over:
        st.plotly_chart(plot_replicates_by_sample(plates), use_container_width=True)

    with t_zoom:
        sample = st.selectbox("Sample", _sample_names(plates), key="rep_zoom_sample")
        d = _processed_df_for_sample(plates, sample)

        fig = px.scatter(
            d,
            x="Time",
            y="baseline_corrected",
            color="key",
            hover_data={"key": True, "plate": True, "well": True, "Time": ":.2f"},
            title=f"Replicates – {sample}",
        )
        fig.update_traces(marker_size=5)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(showgrid=False, title="Time (hours)")
        fig.update_yaxes(showgrid=False, title="OD600 (baseline-corrected)")
        st.plotly_chart(fig, use_container_width=True)


@st.fragment
def ui_window_fits_plate_overview(plates: dict, *, line_hours: float = 4.0):
    plate_id = st.selectbox("Plate", sorted(plates), key="winfit_plate_overview")
    st.plotly_chart(
        plot_window_plate(plates[plate_id], line_hours=line_hours),
        use_container_width=True,
    )


@st.fragment
def ui_window_fits_well_editor(plates: dict, *, line_hours: float = 4.0):
    a, b = st.columns(2)
    plate_id = a.selectbox("Plate", sorted(plates), key="winfit_plate")
    well = b.selectbox("Well", ALL_WELLS, key="winfit_well")

    plate = plates[plate_id]
    key = f"{plate_id}_{well}"

    lag_end, exp_end, no_growth = _phase_controls(plate, well, key=key)
    if no_growth:
        return

    sg_w, sg_p, _ = _sg_params_for_plate(plates, plate_id)
    processed = plate.get("processed_data") or {}
    gs = (plate.get("growth_stats") or {}).get(well) or {}

    fig_d1 = plot_window_single_d1(
        plate, well, sg_window=sg_w, sg_poly=sg_p, frac_peak=0.20
    )
    fig_d2 = plot_window_single_d2(plate, well, sg_window=sg_w, sg_poly=sg_p)

    chart_key = f"lasso_fit_{plate_id}_{well}"
    fig_main = go.Figure(_cached_window_single(processed, well))  # IMPORTANT: copy!
    fig_main = _vlines(
        fig_main, processed, well, lag_end, exp_end, gs=gs, line_hours=line_hours
    )

    st.plotly_chart(
        fig_main,
        key=chart_key,
        selection_mode="lasso",
        on_select=lambda: update_growth_stats_from_lasso(
            plates, plate_id, well, chart_key
        ),
        use_container_width=True,
    )
    st.plotly_chart(fig_d1, use_container_width=True)
    st.plotly_chart(fig_d2, use_container_width=True)


@st.fragment
def ui_growth_summaries(plates: dict):
    rows = []
    for pid, p in plates.items():
        by_name: dict[str, list[str]] = {}
        for well, nm in (p.get("name") or {}).items():
            nm = (nm or "").strip()
            if not nm or nm in ("False", "BLANK"):
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
    order_key = "growth_combined_order"
    st.session_state.setdefault(sel_key, {sid: False for sid in ids})
    st.session_state[sel_key] = {
        sid: st.session_state[sel_key].get(sid, False) for sid in ids
    }
    st.session_state.setdefault(order_key, [])

    max_t = _max_time_hours(plates, default=72.0)

    with st.form("growth_combined_form"):
        t0, t1 = st.slider(
            "Plot time window (hours) — used for Mean growth",
            min_value=0.0,
            max_value=float(max_t),
            value=(0.0, float(min(72.0, max_t))),
            step=0.5,
        )

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

                prev = bool(st.session_state.get(f"{sel_key}:{sid}", False))
                c4.checkbox("", key=f"{sel_key}:{sid}")
                now = bool(st.session_state.get(f"{sel_key}:{sid}", False))
                if now and not prev and sid not in st.session_state[order_key]:
                    st.session_state[order_key].append(sid)

        b1, b2 = st.columns(2)
        apply_stats = b1.form_submit_button(
            "Generate growth stats plot", type="primary", use_container_width=True
        )
        apply_mean = b2.form_submit_button(
            "Generate mean growth plot", type="primary", use_container_width=True
        )

    def _selected_ids_in_order() -> list[str]:
        sel = [
            sid
            for sid in st.session_state[order_key]
            if st.session_state.get(f"{sel_key}:{sid}", False)
        ]
        sel.extend(
            sid
            for sid in ids
            if st.session_state.get(f"{sel_key}:{sid}", False) and sid not in sel
        )
        return sel

    if apply_stats:
        sel_ids = _selected_ids_in_order()
        long_df, sample_order = _build_growth_stats_long_df(plates, sel_ids)
        st.plotly_chart(
            plot_growth_stats(long_df, sample_order), use_container_width=True
        )

    if apply_mean:
        sel_ids = _selected_ids_in_order()
        sel_samples = _unique_preserve_order([sid.split("||", 1)[1] for sid in sel_ids])
        st.plotly_chart(
            plot_mean_growth(plates, sel_samples, t_start=t0, t_end=t1),
            use_container_width=True,
        )


@st.fragment
def ui_export(plates: dict):
    pid = st.selectbox("Plate", sorted(plates), key="export_plate")
    p = plates[pid]
    nm_by_well = p.get("name") or {}

    rows = []
    for well, d in (p.get("processed_data") or {}).items():
        if d is None or d.empty:
            continue
        rows.append(d.assign(plate=pid, well=well, name=(nm_by_well.get(well) or "")))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    st.dataframe(out, use_container_width=True)

    baseline = p.get("baseline")
    if baseline is not None and hasattr(baseline, "empty") and not baseline.empty:
        st.write("Baseline")
        st.dataframe(baseline, use_container_width=True)

    st.download_button(
        "Download processed data CSV",
        out.to_csv(index=False).encode("utf-8"),
        f"{pid}_processed_growth_data.csv",
    )

    gs = (
        pd.DataFrame.from_dict(p.get("growth_stats") or {}, orient="index")
        .rename_axis("well")
        .reset_index()
    )
    gs["Sample Name"] = gs["well"].map(lambda w: (nm_by_well.get(w) or "").strip())
    st.download_button(
        "Download growth stats CSV",
        gs.to_csv(index=False).encode("utf-8"),
        f"{pid}_growth_stats.csv",
    )


# ---------------- Backwards-compatible aliases ----------------
replicates_view = ui_replicates
window_plate_view = ui_window_fits_plate_overview
window_well_view = ui_window_fits_well_editor
growth_stats_view = ui_growth_summaries
export_view = ui_export
