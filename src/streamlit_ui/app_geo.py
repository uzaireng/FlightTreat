"""Module 1 isolated Streamlit workbench."""

from __future__ import annotations

import sys
from pathlib import Path

# Fully initialize pandas before Plotly/Streamlit chart construction.
import pandas as pd  # noqa: F401

import streamlit as st

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from streamlit_ui.geo_workbench import render_geo_workbench

st.set_page_config(
    page_title="CFD SaaS — Geometry Workbench",
    layout="wide",
)

render_geo_workbench()
