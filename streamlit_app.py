"""Multipage entrypoint for the Hakbang PH Streamlit app."""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Hakbang PH · Career Move Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root {
        --night: #030713;
        --panel: #111827;
        --line: rgba(151, 172, 214, .22);
        --text: #f7f8fb;
        --blue: #52a8ff;
        --gold: #ffbd35;
      }
      .stApp {
        background:
          radial-gradient(circle at 78% 3%, rgba(34, 107, 254, .35), transparent 27rem),
          radial-gradient(circle at 10% 40%, rgba(12, 49, 118, .24), transparent 30rem),
          linear-gradient(180deg, #030713 0%, #071022 54%, #050914 100%);
        color: var(--text);
      }
      [data-testid="stHeader"] {background: transparent;}
      [data-testid="stToolbar"] {right: 1rem;}
      [data-testid="stSidebar"] {display: none;}
      .block-container {
        max-width: 1180px;
        padding-top: 1.15rem;
        padding-bottom: 5rem;
      }
      .stApp,
      .stApp [data-testid="stMarkdownContainer"],
      .stApp [data-testid="stMarkdownContainer"] p,
      .stApp [data-testid="stMarkdownContainer"] li,
      .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stWidgetLabel"],
      .stApp [data-testid="stMetricLabel"],
      .stApp [data-testid="stMetricValue"] {
        color: #ffffff !important;
        opacity: 1 !important;
      }
      .page-hero {
        padding: 2rem 2.1rem;
        margin: .5rem 0 1.5rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background:
          radial-gradient(circle at 85% 20%, rgba(45,120,255,.35), transparent 28%),
          linear-gradient(135deg, rgba(16,31,62,.96), rgba(7,13,28,.96));
      }
      .page-hero .eyebrow {
        color: #8fc9ff;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
      }
      .page-hero h1 {
        color: white;
        margin: .45rem 0 .7rem;
        letter-spacing: -.035em;
      }
      .page-hero p {max-width: 850px; margin: 0; line-height: 1.65;}
      div[data-testid="stMetric"] {
        background: rgba(17, 27, 45, .9);
        border: 1px solid var(--line);
        padding: .8rem;
        border-radius: 14px;
      }
      [data-testid="stExpander"] details {
        overflow: hidden;
        border: 1px solid var(--line) !important;
        border-radius: 12px !important;
        background: rgba(17, 27, 45, .92) !important;
      }
      a {color: #77bbff;}
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

career_scan = st.Page(
    "career_scan_page.py",
    title="Career scan",
    icon=":material/travel_explore:",
    default=True,
)
method = st.Page(
    "method_page.py",
    title="Method",
    icon=":material/account_tree:",
    url_path="method",
)
evidence = st.Page(
    "evidence_page.py",
    title="Evidence",
    icon=":material/library_books:",
    url_path="evidence",
)

selected_page = st.navigation(
    [career_scan, method, evidence],
    position="top",
)
selected_page.run()
