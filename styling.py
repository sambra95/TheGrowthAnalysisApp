import streamlit as st


# styles the selected navbar item with green background
def green_navbar():
    return st.markdown(
        """
        <style>
        /* Top navigation - selected/active page */
        button[data-testid="stPageLink-NavLink"][aria-current="page"] {
            background-color: rgba(76, 175, 80, 0.25) !important;
            color: #2e7d32 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #66BB6A !important;
        }

        /* Top navigation - hover effect */
        button[data-testid="stPageLink-NavLink"]:hover {
            background-color: rgba(76, 175, 80, 0.12) !important;
        }

        /* Navigation container styling */
        [data-testid="stSidebarNav"],
        [data-testid="stNavigation"] {
            background-color: transparent !important;
        }

        /* Alternative selectors for active navigation */
        ul[role="tablist"] button[aria-selected="true"],
        div[role="tab"][aria-selected="true"] {
            background-color: rgba(76, 175, 80, 0.25) !important;
            color: #2e7d32 !important;
            font-weight: 600 !important;
            border-bottom: 2px solid #66BB6A !important;
        }

        /* Navigation link active state */
        a[aria-current="page"],
        a[data-active="true"] {
            background-color: rgba(76, 175, 80, 0.25) !important;
            color: #2e7d32 !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# adds a green gradient background at the top of the page
def green_gradient():
    return st.markdown(
        """
        <style>
        /* Green gradient at the top */
        .stMainBlockContainer {
            background: linear-gradient(to bottom, #66BB6A 0%, #FFFFFF 200px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# hacky solution to make all the tertiary buttons red
def red_buttons():

    return st.markdown(
        """
        <style>
        /* Destructive tertiary buttons */
        button[kind="tertiary"] {
            background-color: #ef5350 !important;
            color: white !important;
            border: 1px solid #ef5350 !important;
            border-radius: 0.5rem;
            font-weight: 600;
        }

        button[kind="tertiary"]:hover {
            background-color: #d32f2f !important;
            border-color: #d32f2f !important;
            color: white !important;
        }

        button[kind="tertiary"]:focus {
            box-shadow: 0 0 0 0.2rem rgba(239, 83, 80, 0.3);
            outline: none;
        }

        button[kind="tertiary"]:active {
            background-color: #c62828 !important;
            border-color: #c62828 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
