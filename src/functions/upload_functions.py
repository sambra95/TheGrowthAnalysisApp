"""Data processing helper functions for plate upload and validation."""

from io import BytesIO

import pandas as pd
import streamlit as st

from .constants import BLUE, COLS, DEFAULT_PARAMS, GRAY, GREEN, ORANGE, RED, ROWS


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
    """Build a grid of well status symbols for the plate preview.

    Args:
        plate_map: DataFrame with plate map layout
        present: Set of wells present in the data file
        remove_wells: List/set of wells to exclude, or False
        blank: Whether blank subtraction is enabled

    Returns:
        DataFrame grid with status symbols (GREEN/ORANGE/RED/BLUE/GRAY)
    """
    removed = {w.upper() for w in remove_wells} if remove_wells else set()
    name_by_well = {
        f"{r}{c}": str(plate_map.loc[r, c]).strip() for r in ROWS for c in COLS
    }
    ignored = {w for w, nm in name_by_well.items() if nm == "False"}

    grid = pd.DataFrame(index=ROWS, columns=COLS, dtype="object")
    for r in ROWS:
        for c in COLS:
            w = f"{r}{c}"
            nm = name_by_well.get(w, "")
            is_blank = nm.upper().startswith("BLANK")
            has_valid_name = nm not in {"", "False"} and not nm.upper().startswith("BLANK")

            if w in removed:
                sym = RED
            elif w in ignored:
                sym = GRAY
            elif w not in present:
                sym = ORANGE
            elif blank and is_blank:
                sym = BLUE
            elif has_valid_name:
                sym = GREEN
            else:
                sym = ORANGE

            grid.loc[r, c] = sym
    return grid


def validate_data_file(file_bytes):
    """Validate that the data file has the correct format and data types.

    Args:
        file_bytes: Bytes from uploaded file

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as e:
        return False, f"Failed to read Excel file: {str(e)}"

    if df.empty:
        return False, "Data file is empty"

    # Check for 'Time' column (case-insensitive)
    time_col = None
    for col in df.columns:
        if str(col).strip().lower() == "time":
            time_col = col
            break

    if time_col is None:
        return False, "Data file must contain a 'Time' column"

    # Check if Time column has numeric values
    try:
        time_values = pd.to_numeric(df[time_col], errors="coerce")
        if time_values.isna().all():
            return (
                False,
                "Time column must contain numeric values (integers or decimals)",
            )
        if time_values.isna().any():
            return False, "Time column contains non-numeric values"
    except Exception:
        return False, "Failed to validate Time column data type"

    # Check that there are other columns besides Time (well data)
    if len(df.columns) < 2:
        return False, "Data file must contain well columns in addition to Time column"

    return True, None


def validate_plate_map_file(file_bytes):
    """Validate that the plate map file has the correct format and structure.

    Args:
        file_bytes: Bytes from uploaded file

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    try:
        df = pd.read_excel(BytesIO(file_bytes))
    except Exception as e:
        return False, f"Failed to read Excel file: {str(e)}"

    if df.empty:
        return False, "Plate map file is empty"

    # Check for 'rows' column (case-insensitive)
    rows_col = None
    for col in df.columns:
        if str(col).strip().lower() == "rows":
            rows_col = col
            break

    if rows_col is None:
        return False, "Plate map must contain a 'rows' column"

    # Check that rows column contains expected row labels (A-H)
    expected_rows = set(list("ABCDEFGH"))
    actual_rows = set(df[rows_col].astype(str).str.strip().str.upper())

    if not expected_rows.issubset(actual_rows):
        missing_rows = expected_rows - actual_rows
        return (
            False,
            f"Plate map must contain rows A-H. Missing rows: {', '.join(sorted(missing_rows))}",
        )

    # Check that there are column headers for wells (1-12)
    numeric_cols = []
    for col in df.columns:
        if col != rows_col:
            try:
                col_num = int(str(col).strip())
                if 1 <= col_num <= 12:
                    numeric_cols.append(col_num)
            except (ValueError, TypeError):
                pass

    if len(numeric_cols) < 12:
        return False, "Plate map must contain columns 1-12 for a 96-well plate format"

    return True, None


@st.cache_data(show_spinner="Loading plate preview...")
def get_plate_preview_data(plate_bytes: bytes, data_bytes: bytes):
    """Get plate map and present wells without full analysis.

    This lightweight function only loads the necessary data for the preview
    without running expensive growth curve analysis.

    Args:
        plate_bytes: Bytes of the plate map Excel file
        data_bytes: Bytes of the data Excel file

    Returns:
        tuple: (plate_map DataFrame, set of present wells)
    """
    # Load plate map
    plate_map = pd.read_excel(BytesIO(plate_bytes), index_col=0).fillna("False")

    # Load data and check which wells exist
    data_df = pd.read_excel(BytesIO(data_bytes))

    # Find Time column (case-insensitive)
    time_col = None
    for col in data_df.columns:
        if str(col).strip().lower() == "time":
            time_col = col
            break

    # Present wells are all columns except Time
    present = set(data_df.columns) - {time_col} if time_col else set(data_df.columns)

    return plate_map, present
