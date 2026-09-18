"""Latest pipeline workbench (currently Module 1: Geometry)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # noqa: F401

import streamlit as st

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from streamlit_ui.geo_workbench import render_geo_workbench

st.set_page_config(
    page_title="CFD SaaS — Pipeline Workbench",
    layout="wide",
)

render_geo_workbench()
