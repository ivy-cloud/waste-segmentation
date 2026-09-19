import streamlit as st


def def_css_hitml():
    st.markdown(
        """
        <style>
        .stButton > button {
            border: none;
            color: white;
            padding: 8px 16px;
            text-align: center;
            font-size: 15px;
            margin: 2px 0;
            cursor: pointer;
            border-radius: 8px;
            background-color: #9896f1;
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background-color: #5499C7;
            color: white;
        }
        section[data-testid="stSidebar"] {
            min-width: 280px;
            max-width: 340px;
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {
            white-space: normal !important;
            word-break: keep-all !important;
            overflow-wrap: normal !important;
            width: auto !important;
        }
        div[data-testid="stImage"] img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            width: 100%;
            overflow-x: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
