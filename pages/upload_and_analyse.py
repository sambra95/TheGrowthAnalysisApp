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
ss = init_state()

# Upload (store bytes directly into ss.plates[plate_id]).
u1, u2 = st.columns(2)
with u1:
    with st.container(border=True):
        header_col, popover_col = st.columns([0.85, 0.15])
        with header_col:
            st.header("Step 1. Upload data file")
        with popover_col:
            with st.popover("Help", use_container_width=True):
                st.markdown("**Required Data File Format:**")
                st.markdown("Excel file (.xlsx or .xls) with time series data")
                st.warning(
                    "Do not include a time column. The app will generate time points using 'Read interval' below."
                )

                # Create example data table
                example_data = pd.DataFrame(
                    {
                        "A1": [0.05, 0.08, 0.15, 0.28],
                        "A2": [0.06, 0.09, 0.18, 0.32],
                        "B1": [0.05, 0.07, 0.14, 0.26],
                        "...": ["...", "...", "...", "..."],
                    }
                )
                st.dataframe(example_data, hide_index=True, use_container_width=True)

                # Download example file
                with open("example_data.xlsx", "rb") as f:
                    st.download_button(
                        "Download example data",
                        data=f.read(),
                        file_name="example_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="primary",
                    )
        data_file = st.file_uploader(
            "Plate reader Excel (.xlsx/.xls)", ["xlsx", "xls"], key="data_up"
        )
with u2:
    with st.container(border=True):
        header_col, popover_col = st.columns([0.85, 0.15])
        with header_col:
            st.header("Step 2. Upload plate map")
        with popover_col:
            with st.popover("Help", use_container_width=True):
                st.markdown("**Required Plate Map Format:**")
                st.markdown("Excel file (.xlsx or .xls) with sample layout")
                st.markdown(
                    """
                    - 96-well plate format (rows A-H, columns 1-12)
                    - Samples with the same name will be assigned as replicates
                    - Use 'BLANK' for blank wells
                    - Leave cells empty for wells to ignore
                    - The first '_' is used to split strain and condition labels. These can be used to group samples and colour code with a legend in the 'Create Visulations' page.
                    """
                )

                # Create example plate map table
                example_map = pd.DataFrame(
                    {
                        "rows": ["A", "B", "C", "D"],
                        "1": [
                            "Sample1_Condition1",
                            "Sample3_Condition2",
                            "",
                            "Sample6_Condition3",
                        ],
                        "2": [
                            "Sample1_Condition2",
                            "BLANK",
                            "Sample5_Condition2",
                            "Sample7_Condition2",
                        ],
                        "3": [
                            "Sample2_Condition1",
                            "Sample4_Condition3",
                            "Sample5_Condition2",
                            "BLANK",
                        ],
                        "...": ["...", "...", "...", "..."],
                    }
                )
                st.dataframe(example_map, hide_index=True, use_container_width=True)

                # Download example file
                with open("example_plate_map.xls", "rb") as f:
                    st.download_button(
                        "Download example plate map",
                        data=f.read(),
                        file_name="example_plate_map.xls",
                        mime="application/vnd.ms-excel",
                        use_container_width=True,
                        type="primary",
                    )
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


with st.container(border=True):
    st.header("Step 3. Select the analysis parameters")

    ready = sorted(ss.plates)

    pcol, acol = st.columns(2, gap="large")

    with pcol:
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

                st.write("")
                st.caption("Plate preview:")

                st.caption(
                    "· 🟩 analyzable · 🟧 missing data · 🟥 excluded · 🟦 blank · ⬜ not in plate map"
                )
                render_plate_table(grid)

        else:
            st.warning("Upload files for to see plate preview.")

    col1, col2 = st.columns(2)
    if col1.button(
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

    if col2.button(
        "Remove selected plate",
        type="tertiary",
        use_container_width=True,
        disabled=not plate_id,
    ):
        ss.plates.pop(plate_id, None)
        st.rerun()
