"""High-level plate overview panels used by Streamlit pages."""
import streamlit as st

from functions.plotting_functions import (
    plot_baseline,
    plot_replicates_by_sample,
    plot_window_plate,
)


# ---------------- Gatekeeper ----------------
def require_plates() -> dict:
    """Return plates from session state, or stop with a warning."""
    plates = st.session_state.get("plates") or {}
    if not plates:
        st.warning("No results yet. Run **Upload + Analyse** first.")
        st.stop()
    return plates


# ---------------- Fragments used by pages ----------------
@st.fragment
def ui_replicates(plates: dict):
    """Render a grid of replicate plots by sample."""
    st.subheader("Sample Replicates")
    st.plotly_chart(plot_replicates_by_sample(plates), width='stretch')


@st.fragment
def ui_window_fits_plate_overview(plates: dict):
    """Render baseline and plate-window fits for a selected plate."""
    plate_id = st.selectbox("Plate", sorted(plates), key="winfit_plate_overview")
    st.subheader("Plate Blanks")
    plate = plates[plate_id]
    st.plotly_chart(
        plot_baseline(plate["baseline"], name_by_well=plate.get("name", {}))
    )

    st.subheader("Plate Fits Overview")
    st.plotly_chart(
        plot_window_plate(plates[plate_id]),
        width='stretch',
    )


# ---------------- Backwards-compatible aliases ----------------
replicates_view = ui_replicates
window_plate_view = ui_window_fits_plate_overview
