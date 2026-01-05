"""Upload inputs, configure analysis parameters, and run plate analysis."""
import pandas as pd
import streamlit as st

from functions.data_processing import analyse_plate, load_plate

ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))

GREEN = "🟩"
ORANGE = "🟧"
RED = "🟥"
BLUE = "🟦"
GRAY = "⬜"

DEFAULT_PARAMS = dict(
    read_interval_min=12,
    pathlength_cm_=0.42,
    clip_time_series=(0.0, 72.0),
    remove_wells=False,
    blank=True,
    window_points=15,
    sg_window=15,
    sg_poly=2,
)


def init_state():
    """Ensure required session state keys exist."""
    st.session_state.setdefault("plates", {})
    return st.session_state


def plate_params(ss, plate_id: str) -> dict:
    """Return stored params for a plate or the defaults."""
    return (ss.plates.get(plate_id, {}) or {}).get("params", DEFAULT_PARAMS)


def build_symbol_grid(
    *, plate_map: pd.DataFrame, present: set[str], remove_wells=False, blank=True
):
    """Build a grid of well status symbols for the plate preview."""
    removed = {w.upper() for w in remove_wells} if remove_wells else set()
    name_by_well = {f"{r}{c}": str(plate_map.loc[r, c]) for r in ROWS for c in COLS}
    ignored = {w for w, nm in name_by_well.items() if nm == "False"}

    grid = pd.DataFrame(index=ROWS, columns=COLS, dtype="object")
    for r in ROWS:
        for c in COLS:
            w = f"{r}{c}"
            nm = name_by_well.get(w, "")
            is_blank = nm == "BLANK"

            if w in removed:
                sym = RED
            elif w in present:
                sym = GREEN
            elif blank and is_blank:
                sym = BLUE
            elif w in ignored:
                sym = GRAY
            else:
                sym = ORANGE

            grid.loc[r, c] = sym
    return grid


def render_plate_table(grid: pd.DataFrame):
    """Render an HTML table showing the plate status grid."""
    css = """
    <style>
      .plate-wrap { width: 100%; overflow: hidden; }
      table.plate {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 18px;
      }
      table.plate th, table.plate td {
        border: 1px solid rgba(49,51,63,0.2);
        text-align: center;
        padding: 6px 0;
        line-height: 1.2;
      }
      table.plate th { font-weight: 600; }
      table.plate th.row { width: 2.2rem; }
    </style>
    """

    header = "".join(f"<th>{c}</th>" for c in grid.columns)
    rows_html = []
    for r in grid.index:
        cells = "".join(f"<td>{grid.loc[r, c]}</td>" for c in grid.columns)
        rows_html.append(f"<tr><th class='row'>{r}</th>{cells}</tr>")

    html = f"""
    {css}
    <div class="plate-wrap">
      <table class="plate">
        <thead>
          <tr><th></th>{header}</tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


# ---------------- App ----------------
st.title("1) Upload a datafile and a plate map")

ss = init_state()

# Upload (store bytes directly into ss.plates[plate_id]).
u1, u2 = st.columns(2)
with u1:
    data_file = st.file_uploader(
        "Plate reader Excel (.xlsx/.xls)", ["xlsx", "xls"], key="data_up"
    )
with u2:
    map_file = st.file_uploader(
        "Plate map (.xls/.xlsx) with 'rows' column", ["xlsx", "xls"], key="map_up"
    )

if st.button(
    "Load plate",
    type="primary",
    use_container_width=True,
    disabled=not (data_file and map_file),
):
    plate_id = (
        data_file.name.rsplit(".", 1)[0]
        if getattr(data_file, "name", None)
        else "Plate"
    )
    load_plate(
        ss.plates,
        plate_id,
        data_bytes=data_file.getvalue(),
        plate_bytes=map_file.getvalue(),
        params=DEFAULT_PARAMS,
    )
    st.toast(f"Saved uploads for {plate_id}")

st.divider()
st.title("2) Select the analysis parameters")

ready = sorted(ss.plates)

pcol, acol = st.columns(2, gap="large")

with pcol:
    st.subheader("Analysis Parameters")
    st.divider()
    plate_id = st.selectbox("Plate to analyse", ready, disabled=not ready)
    params0 = plate_params(ss, plate_id) if plate_id else DEFAULT_PARAMS

    a, b = st.columns(2)
    read_interval_min = a.number_input(
        "Read interval (min)", 1, 120, int(params0["read_interval_min"])
    )
    pl_cm = b.number_input(
        "Pathlength (cm)",
        value=float(params0["pathlength_cm_"]),
        step=0.01,
        format="%.3f",
    )

    a, b = st.columns(2)
    clip_time_series = (
        float(
            a.number_input(
                "Start (h)", 0.0, 1e6, float(params0["clip_time_series"][0]), 0.5
            )
        ),
        float(
            b.number_input(
                "End (h)", 0.0, 1e6, float(params0["clip_time_series"][1]), 0.5
            )
        ),
    )

    blank = a.checkbox("Blank subtraction (label 'BLANK')", bool(params0["blank"]))

    remove_wells = st.multiselect(
        "Exclude wells",
        options=[f"{r}{c}" for r in "ABCDEFGH" for c in range(1, 13)],
        default=[],
    )

    # Preserve the False sentinel behavior used elsewhere.
    remove_wells = remove_wells if remove_wells else False

    window_points = st.number_input(
        "Window size for maximum growth rate (points)",
        5,
        200,
        int(params0["window_points"]),
        1,
    )

with acol:
    st.subheader("Plate Overview")

    params = dict(
        read_interval_min=int(read_interval_min),
        pathlength_cm_=float(pl_cm),
        clip_time_series=clip_time_series,
        remove_wells=remove_wells,
        blank=bool(blank),
        window_points=int(window_points),
        sg_window=int(params0.get("sg_window", 15)),
        sg_poly=int(params0.get("sg_poly", 2)),
    )

    st.divider()

    # Preview grid.
    if plate_id:
        rec = ss.plates.get(plate_id, {})
        if rec.get("uploads"):
            tmp = {"uploads": rec["uploads"], "params": params}  # params from UI
            plate_preview = analyse_plate(tmp)
            present = set(plate_preview.get("growth_stats", {}).keys())

            grid = build_symbol_grid(
                plate_map=plate_preview["plate_map"],
                present=present,
                remove_wells=params["remove_wells"],
                blank=params["blank"],
            )
            st.caption(
                "· 🟩 analyzable · 🟧 missing data · 🟥 excluded · 🟦 blank · ⬜ not in plate map"
            )
            render_plate_table(grid)

            st.write("")
            st.write("")

            if st.button(
                "Remove selected plate",
                type="tertiary",
                use_container_width=True,
                disabled=not plate_id,
            ):
                ss.plates.pop(plate_id, None)
                st.rerun()

    else:
        st.warning("Upload files for to see plate preview.")

st.divider()

st.title("3) Click analyse")

if st.button(
    "Analyse selected plate",
    type="primary",
    use_container_width=True,
    disabled=not plate_id,
):
    rec = ss.plates.get(plate_id, {})
    if not rec.get("uploads"):
        st.error("No uploads found for this plate.")
    else:
        rec["params"] = params
        ss.plates[plate_id] = analyse_plate(rec)
        st.toast(f"Analysed {plate_id}", duration="infinite")
