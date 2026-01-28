import streamlit as st


# adds a green gradient background at the top of the page
def green_gradient():
    return st.markdown(
        """
        <style>
        /* Green gradient at the top */
        .stMainBlockContainer {
            background: linear-gradient(to bottom, #4CAF50 0%, #FFFFFF 200px);
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
            background-color: #d32f2f !important;
            color: white !important;
            border: 1px solid #d32f2f !important;
            border-radius: 0.5rem;
            font-weight: 600;
        }

        button[kind="tertiary"]:hover {
            background-color: #b71c1c !important;
            border-color: #b71c1c !important;
            color: white !important;
        }

        button[kind="tertiary"]:focus {
            box-shadow: 0 0 0 0.2rem rgba(211, 47, 47, 0.4);
            outline: none;
        }

        button[kind="tertiary"]:active {
            background-color: #8e0000 !important;
            border-color: #8e0000 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
