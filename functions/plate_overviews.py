# functions/analysis_shared.py
import streamlit as st

from functions.plotting_functions import (
    plot_replicates_by_sample,
    plot_window_plate,
    plot_baseline,
)


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Fragments used by pages ----------------
@st.fragment
def ui_replicates(plates: dict):
    st.subheader("Sample Replicates")
    st.plotly_chart(plot_replicates_by_sample(plates), use_container_width=True)


@st.fragment
def ui_window_fits_plate_overview(plates: dict, *, line_hours: float = 4.0):

    plate_id = st.selectbox("Plate", sorted(plates), key="winfit_plate_overview")
    st.subheader("Plate Blanks")
    st.plotly_chart(plot_baseline(plates[plate_id]["baseline"]))

    st.subheader("Plate Fits Overview")
    st.plotly_chart(
        plot_window_plate(plates[plate_id], line_hours=line_hours),
        use_container_width=True,
    )


# ---------------- Backwards-compatible aliases ----------------
replicates_view = ui_replicates
window_plate_view = ui_window_fits_plate_overview
