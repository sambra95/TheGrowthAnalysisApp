import streamlit as st
from st_selectable_grid import st_selectable_grid

ROWS = 8
COLS = 12
DEFAULT_GROUP = "Group 1"

GROUP_PALETTE = [
    "#dbeafe",
    "#dcfce7",
    "#fde68a",
    "#fbcfe8",
    "#e9d5ff",
    "#fed7aa",
    "#bfdbfe",
    "#ddd6fe",
]


# -----------------------------
# Helpers
# -----------------------------
def make_assignments(fill_value=DEFAULT_GROUP):
    return [[fill_value for _ in range(COLS)] for _ in range(ROWS)]


def fill_rect(assignments, p1, p2, value):
    x1, y1 = p1["x"], p1["y"]
    x2, y2 = p2["x"], p2["y"]
    left, right = sorted([x1, x2])
    top, bottom = sorted([y1, y2])

    new_grid = [row[:] for row in assignments]
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            new_grid[r][c] = value
    return new_grid


def reassign_group(assignments, from_group, to_group=DEFAULT_GROUP):
    new_grid = [row[:] for row in assignments]
    for r in range(ROWS):
        for c in range(COLS):
            if new_grid[r][c] == from_group:
                new_grid[r][c] = to_group
    return new_grid


def build_cells(assignments, color_map):
    cells = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            group = assignments[r][c]
            label = f"{chr(65 + r)}{c + 1}"
            row.append(
                {
                    "label": label,
                    "cell_color": color_map.get(group, "#ffffff"),
                    "tooltip": f"{label} · {group}",
                }
            )
        cells.append(row)
    return cells


def rect_from_points(p1, p2):
    left, right = sorted([p1["x"], p2["x"]])
    top, bottom = sorted([p1["y"], p2["y"]])
    return top, bottom, left, right


def reset_pending_selection():
    st.session_state.first_corner = None
    st.session_state.awaiting_second = False
    st.session_state.pending_clear_mode = None


def get_effective_clear_mode():
    if (
        st.session_state.awaiting_second
        and st.session_state.pending_clear_mode is not None
    ):
        return st.session_state.pending_clear_mode
    return False


def group_number(name):
    return int(name.replace("Group ", ""))


def next_group_name(groups):
    nums = [group_number(g) for g in groups]
    return f"Group {max(nums) + 1}" if nums else "Group 1"


def color_for_group(group_name):
    n = group_number(group_name)
    return GROUP_PALETTE[(n - 1) % len(GROUP_PALETTE)]


# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="Well Plate Group Assigner", layout="wide")
st.title("Well plate group assigner")

# -----------------------------
# Session state
# -----------------------------
if "groups" not in st.session_state:
    st.session_state.groups = [DEFAULT_GROUP]

if "group_colors" not in st.session_state:
    st.session_state.group_colors = {
        group: color_for_group(group) for group in st.session_state.groups
    }

if "assignments" not in st.session_state:
    st.session_state.assignments = make_assignments(DEFAULT_GROUP)

if "active_group" not in st.session_state:
    st.session_state.active_group = DEFAULT_GROUP

if "first_corner" not in st.session_state:
    st.session_state.first_corner = None

if "awaiting_second" not in st.session_state:
    st.session_state.awaiting_second = False

if "consumed_click" not in st.session_state:
    st.session_state.consumed_click = None

if "pending_clear_mode" not in st.session_state:
    st.session_state.pending_clear_mode = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.subheader("Groups")

    current_index = st.session_state.groups.index(st.session_state.active_group)
    selected_group = st.selectbox(
        "Assigned group",
        st.session_state.groups,
        index=current_index,
    )

    if selected_group != st.session_state.active_group:
        st.session_state.active_group = selected_group
        reset_pending_selection()

    assigned_color = st.session_state.group_colors[st.session_state.active_group]
    st.markdown(
        f"**Assigned color:** "
        f"<span style='display:inline-block;width:14px;height:14px;"
        f"background:{assigned_color};border:1px solid #999;vertical-align:middle;'></span>",
        unsafe_allow_html=True,
    )

    if st.button("Add group", use_container_width=True):
        reset_pending_selection()
        new_group = next_group_name(st.session_state.groups)
        st.session_state.groups.append(new_group)
        st.session_state.group_colors[new_group] = color_for_group(new_group)
        st.session_state.active_group = new_group
        st.session_state.consumed_click = None
        st.rerun()

    last_group = st.session_state.groups[-1]
    can_delete = last_group != DEFAULT_GROUP

    if st.button(
        "Delete last group", use_container_width=True, disabled=not can_delete
    ):
        reset_pending_selection()
        group_to_delete = st.session_state.groups[-1]

        st.session_state.groups.pop()
        st.session_state.assignments = reassign_group(
            st.session_state.assignments, group_to_delete, DEFAULT_GROUP
        )
        st.session_state.group_colors.pop(group_to_delete)

        if st.session_state.active_group == group_to_delete:
            st.session_state.active_group = DEFAULT_GROUP

        st.session_state.consumed_click = None
        st.rerun()

    if not can_delete:
        st.caption("Group 1 cannot be deleted.")


# -----------------------------
# Instructions
# -----------------------------
effective_clear_mode = get_effective_clear_mode()
mode_label = (
    f"reassign to {DEFAULT_GROUP}"
    if effective_clear_mode
    else f"assign to {st.session_state.active_group}"
)

st.write(
    f"Click one well, then click a second well. "
    f"On the second click, the full rectangle is applied to **{mode_label}**."
)
st.caption(
    "Tip: click the same well twice to apply to a single well. "
    "If the first clicked well is already in the selected group, "
    f"the action switches to **{DEFAULT_GROUP}**."
)


# -----------------------------
# Grid
# -----------------------------
header = [str(i + 1) for i in range(COLS)]
index = [chr(65 + i) for i in range(ROWS)]

cells = build_cells(
    st.session_state.assignments,
    st.session_state.group_colors,
)

selection = st_selectable_grid(
    cells=cells,
    header=header,
    index=index,
    aspect_ratio=1.0,
    allow_secondary_selection=False,
    allow_header_selection=False,
    resize=True,
    height=560,
    primary_selection_color="#2563eb",
    key="well_grid",
)

current_click = selection.get("primary") if selection else None

# -----------------------------
# Click handling
# -----------------------------
if not st.session_state.awaiting_second:
    if current_click:
        click_key = (current_click["x"], current_click["y"])
        if click_key != st.session_state.consumed_click:
            x, y = current_click["x"], current_click["y"]
            clicked_group = st.session_state.assignments[y][x]

            auto_clear = clicked_group == st.session_state.active_group

            st.session_state.pending_clear_mode = auto_clear
            st.session_state.first_corner = current_click
            st.session_state.awaiting_second = True
            st.session_state.consumed_click = click_key

else:
    first = st.session_state.first_corner
    effective_clear_mode = get_effective_clear_mode()
    value = DEFAULT_GROUP if effective_clear_mode else st.session_state.active_group

    if current_click is None:
        st.session_state.assignments = fill_rect(
            st.session_state.assignments,
            first,
            first,
            value,
        )
        reset_pending_selection()
        st.session_state.consumed_click = None
        st.rerun()

    else:
        click_key = (current_click["x"], current_click["y"])
        first_key = (first["x"], first["y"])

        if click_key != first_key and click_key != st.session_state.consumed_click:
            st.session_state.assignments = fill_rect(
                st.session_state.assignments,
                first,
                current_click,
                value,
            )
            reset_pending_selection()
            st.session_state.consumed_click = click_key
            st.rerun()
