import streamlit as st
from st_selectable_grid import st_selectable_grid

ROWS, COLS = 8, 12
DEFAULT_GROUP = "Group 1"
GROUP_PALETTE = [
    "#ccebc5",  # mint
    "#8dd3c7",  # soft teal
    "#ffffb3",  # soft yellow
    "#bebada",  # lavender
    "#fb8072",  # muted coral
    "#80b1d3",  # dusty blue
    "#fdb462",  # soft orange
    "#b3de69",  # yellow-green
    "#fccde5",  # pink
    "#bc80bd",  # muted purple
    "#ffed6f",  # warm yellow
]


def group_number(name):
    return int(name.replace("Group ", ""))


def next_group_name(groups):
    return (
        f"Group {max(group_number(g) for g in groups) + 1}" if groups else DEFAULT_GROUP
    )


def color_for_group(group_name):
    return GROUP_PALETTE[(group_number(group_name) - 1) % len(GROUP_PALETTE)]


def make_assignments(fill_value=DEFAULT_GROUP):
    return [[fill_value for _ in range(COLS)] for _ in range(ROWS)]


def fill_rect(assignments, p1, p2, value):
    left, right = sorted((p1["x"], p2["x"]))
    top, bottom = sorted((p1["y"], p2["y"]))
    new_grid = [row[:] for row in assignments]
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            new_grid[r][c] = value
    return new_grid


def reassign_group(assignments, from_group, to_group=DEFAULT_GROUP):
    return [
        [to_group if cell == from_group else cell for cell in row]
        for row in assignments
    ]


def build_cells(assignments, color_map):
    return [
        [
            {
                "label": f"{chr(65 + r)}{c + 1}",
                "cell_color": color_map.get(group, "#ffffff"),
                "tooltip": f"{chr(65 + r)}{c + 1} · {group}",
            }
            for c, group in enumerate(row)
        ]
        for r, row in enumerate(assignments)
    ]


def reset_pending_selection():
    ss = st.session_state
    ss.first_corner = None
    ss.awaiting_second = False
    ss.pending_clear_mode = None


def init_state():
    ss = st.session_state
    ss.setdefault("groups", [DEFAULT_GROUP])
    ss.setdefault("group_colors", {g: color_for_group(g) for g in ss.groups})
    ss.setdefault("assignments", make_assignments(DEFAULT_GROUP))
    ss.setdefault("active_group", DEFAULT_GROUP)
    ss.setdefault("first_corner", None)
    ss.setdefault("awaiting_second", False)
    ss.setdefault("consumed_click", None)
    ss.setdefault("pending_clear_mode", None)


st.set_page_config(page_title="Well Plate Group Assigner", layout="wide")
st.title("Well plate group assigner")
init_state()
ss = st.session_state

controls_col, plate_col = st.columns([1, 4])

with controls_col:
    st.subheader("Groups")
    selected_group = st.selectbox(
        "Assigned group",
        ss.groups,
        index=ss.groups.index(ss.active_group),
    )
    if selected_group != ss.active_group:
        ss.active_group = selected_group
        reset_pending_selection()

    assigned_color = ss.group_colors[ss.active_group]
    st.markdown(
        f"**Assigned color:** "
        f"<span style='display:inline-block;width:14px;height:14px;"
        f"background:{assigned_color};border:1px solid #999;vertical-align:middle;'></span>",
        unsafe_allow_html=True,
    )

    add_col, delete_col = st.columns(2)
    add_group_clicked = add_col.button("Add group", use_container_width=True)
    delete_group_clicked = delete_col.button(
        "Remove group",
        use_container_width=True,
        disabled=ss.active_group == DEFAULT_GROUP,
    )

    if add_group_clicked:
        reset_pending_selection()
        new_group = next_group_name(ss.groups)
        ss.groups.append(new_group)
        ss.group_colors[new_group] = color_for_group(new_group)
        ss.active_group = new_group
        ss.consumed_click = None
        st.rerun()

    if delete_group_clicked:
        reset_pending_selection()
        group_to_delete = ss.active_group
        ss.groups.remove(group_to_delete)
        ss.assignments = reassign_group(ss.assignments, group_to_delete, DEFAULT_GROUP)
        ss.group_colors.pop(group_to_delete, None)
        ss.active_group = DEFAULT_GROUP
        ss.consumed_click = None
        st.rerun()

with plate_col:
    selection = st_selectable_grid(
        cells=build_cells(ss.assignments, ss.group_colors),
        header=[str(i + 1) for i in range(COLS)],
        index=[chr(65 + i) for i in range(ROWS)],
        aspect_ratio=1.0,
        allow_secondary_selection=False,
        allow_header_selection=False,
        resize=True,
        height=560,
        primary_selection_color="#2563eb",
        key="well_grid",
    )

current_click = (selection or {}).get("primary")

if not ss.awaiting_second:
    if current_click:
        click_key = (current_click["x"], current_click["y"])
        if click_key != ss.consumed_click:
            x, y = click_key
            ss.pending_clear_mode = ss.assignments[y][x] == ss.active_group
            ss.first_corner = current_click
            ss.awaiting_second = True
            ss.consumed_click = click_key
else:
    first = ss.first_corner
    value = DEFAULT_GROUP if ss.pending_clear_mode else ss.active_group

    if current_click is None:
        ss.assignments = fill_rect(ss.assignments, first, first, value)
        reset_pending_selection()
        ss.consumed_click = None
        st.rerun()
    else:
        click_key = (current_click["x"], current_click["y"])
        first_key = (first["x"], first["y"])
        if click_key not in (first_key, ss.consumed_click):
            ss.assignments = fill_rect(ss.assignments, first, current_click, value)
            reset_pending_selection()
            ss.consumed_click = click_key
            st.rerun()
