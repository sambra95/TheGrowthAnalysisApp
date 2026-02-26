"""UI helpers for assigning blank-subtraction analysis groups."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st

from src.functions.constants import COLS, ROWS

try:
    from st_selectable_grid import st_selectable_grid
except ImportError:  # pragma: no cover - optional dependency
    st_selectable_grid = None


DEFAULT_GROUP = "Group 1"
GROUP_PALETTE = [
    "#ccebc5",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
    "#ffed6f",
]


def _state_key(prefix: str, suffix: str) -> str:
    return f"{prefix}::{suffix}"


def group_number(name: str) -> int:
    """Extract numeric suffix from group name."""
    match = re.search(r"(\d+)", str(name))
    return int(match.group(1)) if match else 1


def _group_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(name))
    if match:
        return int(match.group(1)), str(name)
    return 10**9, str(name)


def next_group_name(groups: list[str]) -> str:
    return f"Group {max(group_number(g) for g in groups) + 1}" if groups else DEFAULT_GROUP


def color_for_group(group_name: str) -> str:
    return GROUP_PALETTE[(group_number(group_name) - 1) % len(GROUP_PALETTE)]


def make_assignments(fill_value: str = DEFAULT_GROUP) -> list[list[str]]:
    return [[fill_value for _ in COLS] for _ in ROWS]


def fill_rect(
    assignments: list[list[str]], p1: dict[str, int], p2: dict[str, int], value: str
) -> list[list[str]]:
    left, right = sorted((p1["x"], p2["x"]))
    top, bottom = sorted((p1["y"], p2["y"]))
    out = [row[:] for row in assignments]
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            out[r][c] = value
    return out


def reassign_group(
    assignments: list[list[str]], from_group: str, to_group: str = DEFAULT_GROUP
) -> list[list[str]]:
    return [[to_group if cell == from_group else cell for cell in row] for row in assignments]


def _well_to_point(well: str) -> dict[str, int] | None:
    text = str(well).strip().upper()
    if len(text) < 2:
        return None
    row = text[0]
    col = text[1:]
    if row not in ROWS:
        return None
    try:
        col_num = int(col)
    except ValueError:
        return None
    if col_num not in COLS:
        return None
    return {"x": COLS.index(col_num), "y": ROWS.index(row)}


def _point_to_well(point: dict[str, int]) -> str:
    return f"{ROWS[point['y']]}{COLS[point['x']]}"


def _assignments_from_map(group_map: dict[str, str] | None) -> list[list[str]]:
    assignments = make_assignments(DEFAULT_GROUP)
    if not isinstance(group_map, dict):
        return assignments

    for well, group_name in group_map.items():
        point = _well_to_point(str(well))
        if point is None:
            continue
        normalized = str(group_name).strip() or DEFAULT_GROUP
        assignments[point["y"]][point["x"]] = normalized
    return assignments


def assignments_to_map(assignments: list[list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for y, row_label in enumerate(ROWS):
        for x, col_label in enumerate(COLS):
            out[f"{row_label}{col_label}"] = str(assignments[y][x])
    return out


def _group_names_from_assignments(assignments: list[list[str]]) -> list[str]:
    groups = sorted(
        {str(cell).strip() or DEFAULT_GROUP for row in assignments for cell in row},
        key=_group_sort_key,
    )
    if DEFAULT_GROUP not in groups:
        groups.insert(0, DEFAULT_GROUP)
    return groups


def build_cells(
    assignments: list[list[str]],
    color_map: dict[str, str],
    name_by_well: dict[str, str] | None = None,
) -> list[list[dict[str, Any]]]:
    name_by_well = name_by_well or {}
    rows: list[list[dict[str, Any]]] = []
    for y, row in enumerate(assignments):
        rendered_row = []
        for x, group in enumerate(row):
            well = f"{ROWS[y]}{COLS[x]}"
            sample = str(name_by_well.get(well, "")).strip()
            sample_suffix = f" · {sample}" if sample and sample not in {"False"} else ""
            rendered_row.append(
                {
                    "label": well,
                    "cell_color": color_map.get(group, "#ffffff"),
                    "tooltip": f"{well}{sample_suffix} · {group}",
                }
            )
        rows.append(rendered_row)
    return rows


def _reset_pending_selection(prefix: str):
    ss = st.session_state
    ss[_state_key(prefix, "first_corner")] = None
    ss[_state_key(prefix, "awaiting_second")] = False
    ss[_state_key(prefix, "pending_clear_mode")] = None


def _init_state(prefix: str, initial_group_map: dict[str, str] | None):
    ss = st.session_state
    assignments_key = _state_key(prefix, "assignments")
    groups_key = _state_key(prefix, "groups")
    colors_key = _state_key(prefix, "group_colors")
    active_key = _state_key(prefix, "active_group")

    if assignments_key not in ss:
        ss[assignments_key] = _assignments_from_map(initial_group_map)

    if groups_key not in ss:
        ss[groups_key] = _group_names_from_assignments(ss[assignments_key])
    elif not ss[groups_key]:
        ss[groups_key] = [DEFAULT_GROUP]

    if colors_key not in ss:
        ss[colors_key] = {g: color_for_group(g) for g in ss[groups_key]}
    else:
        for g in ss[groups_key]:
            ss[colors_key].setdefault(g, color_for_group(g))

    if active_key not in ss or ss[active_key] not in ss[groups_key]:
        ss[active_key] = DEFAULT_GROUP if DEFAULT_GROUP in ss[groups_key] else ss[groups_key][0]

    ss.setdefault(_state_key(prefix, "first_corner"), None)
    ss.setdefault(_state_key(prefix, "awaiting_second"), False)
    ss.setdefault(_state_key(prefix, "consumed_click"), None)
    ss.setdefault(_state_key(prefix, "pending_clear_mode"), None)


def _render_fallback_assigner(prefix: str):
    ss = st.session_state
    assignments_key = _state_key(prefix, "assignments")
    active_key = _state_key(prefix, "active_group")

    st.caption(
        "`st_selectable_grid` is not installed. Using range controls for group assignment."
    )

    all_wells = [f"{r}{c}" for r in ROWS for c in COLS]
    c1, c2 = st.columns(2)
    start_well = c1.selectbox("Start well", all_wells, key=_state_key(prefix, "range_start"))
    end_well = c2.selectbox("End well", all_wells, key=_state_key(prefix, "range_end"))
    p1 = _well_to_point(start_well) or {"x": 0, "y": 0}
    p2 = _well_to_point(end_well) or {"x": 0, "y": 0}

    a, b = st.columns(2)
    if a.button(
        "Assign range",
        type="primary",
        width="stretch",
        key=_state_key(prefix, "assign_range"),
    ):
        ss[assignments_key] = fill_rect(
            ss[assignments_key], p1, p2, ss[active_key]
        )
        st.rerun()
    if b.button(
        "Clear range",
        type="primary",
        width="stretch",
        key=_state_key(prefix, "clear_range"),
    ):
        ss[assignments_key] = fill_rect(
            ss[assignments_key], p1, p2, DEFAULT_GROUP
        )
        st.rerun()

    df = pd.DataFrame(ss[assignments_key], index=ROWS, columns=COLS)
    st.dataframe(df, width="stretch", height=350)


def ui_blank_group_assigner(
    *,
    plate_id: str,
    initial_group_map: dict[str, str] | None = None,
    name_by_well: dict[str, str] | None = None,
) -> dict[str, str]:
    """Render the analysis-group assignment UI and return well->group mapping."""
    prefix = f"blank_groups::{plate_id}"
    _init_state(prefix, initial_group_map)
    ss = st.session_state

    groups_key = _state_key(prefix, "groups")
    colors_key = _state_key(prefix, "group_colors")
    assignments_key = _state_key(prefix, "assignments")
    active_key = _state_key(prefix, "active_group")
    awaiting_key = _state_key(prefix, "awaiting_second")
    first_key = _state_key(prefix, "first_corner")
    consumed_key = _state_key(prefix, "consumed_click")
    clear_mode_key = _state_key(prefix, "pending_clear_mode")

    st.caption(
        "Assign plate wells to analysis groups. Each sample well is blank-subtracted "
        "using BLANK wells from the same group only."
    )

    group_col, color_col, add_col, remove_col = st.columns(
        [2.5, 0.75, 1.5, 1.5], vertical_alignment="center"
    )
    selected_group = group_col.selectbox(
        "Assigned group",
        ss[groups_key],
        index=ss[groups_key].index(ss[active_key]),
        key=_state_key(prefix, "assigned_group_select"),
    )
    if selected_group != ss[active_key]:
        ss[active_key] = selected_group
        _reset_pending_selection(prefix)

    with color_col:
        assigned_color = ss[colors_key][ss[active_key]]
        st.markdown(
            (
                "<span style='display:inline-block;width:22px;height:22px;"
                f"background:{assigned_color};border:1px solid #999;'></span>"
            ),
            unsafe_allow_html=True,
        )

    add_clicked = add_col.button(
        "Add group",
        type="primary",
        width="stretch",
        key=_state_key(prefix, "add_group"),
    )
    remove_clicked = remove_col.button(
        "Remove group",
        type="primary",
        width="stretch",
        key=_state_key(prefix, "remove_group"),
        disabled=ss[active_key] == DEFAULT_GROUP,
    )

    if add_clicked:
        _reset_pending_selection(prefix)
        new_group = next_group_name(ss[groups_key])
        ss[groups_key].append(new_group)
        ss[colors_key][new_group] = color_for_group(new_group)
        ss[active_key] = new_group
        ss[consumed_key] = None
        st.rerun()

    if remove_clicked:
        _reset_pending_selection(prefix)
        remove_group = ss[active_key]
        ss[groups_key].remove(remove_group)
        ss[assignments_key] = reassign_group(
            ss[assignments_key], remove_group, DEFAULT_GROUP
        )
        ss[colors_key].pop(remove_group, None)
        ss[active_key] = DEFAULT_GROUP
        ss[consumed_key] = None
        st.rerun()

    if st_selectable_grid is None:
        _render_fallback_assigner(prefix)
    else:
        selection = st_selectable_grid(
            cells=build_cells(ss[assignments_key], ss[colors_key], name_by_well),
            header=[str(c) for c in COLS],
            index=ROWS,
            aspect_ratio=1.0,
            allow_secondary_selection=False,
            allow_header_selection=False,
            resize=True,
            height=460,
            primary_selection_color="#2563eb",
            key=_state_key(prefix, "well_grid"),
        )

        current_click = (selection or {}).get("primary")

        if not ss[awaiting_key]:
            if current_click:
                click_key = (current_click["x"], current_click["y"])
                if click_key != ss[consumed_key]:
                    x, y = click_key
                    ss[clear_mode_key] = ss[assignments_key][y][x] == ss[active_key]
                    ss[first_key] = current_click
                    ss[awaiting_key] = True
                    ss[consumed_key] = click_key
        else:
            first_corner = ss[first_key]
            value = DEFAULT_GROUP if ss[clear_mode_key] else ss[active_key]

            if current_click is None:
                ss[assignments_key] = fill_rect(
                    ss[assignments_key], first_corner, first_corner, value
                )
                _reset_pending_selection(prefix)
                ss[consumed_key] = None
                st.rerun()
            else:
                click_key = (current_click["x"], current_click["y"])
                first_corner_key = (first_corner["x"], first_corner["y"])
                if click_key not in (first_corner_key, ss[consumed_key]):
                    ss[assignments_key] = fill_rect(
                        ss[assignments_key], first_corner, current_click, value
                    )
                    _reset_pending_selection(prefix)
                    ss[consumed_key] = click_key
                    st.rerun()

    assignment_map = assignments_to_map(ss[assignments_key])
    return assignment_map
