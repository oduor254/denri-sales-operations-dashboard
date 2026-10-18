"""Denri Sales Operations Dashboard — for the sales manager.

Five pages: Sales, Shop Attendants, Shop Performance, WhatsApp & Social DMs,
live from Postgres (Shop Attendants also reads feedback and training scores
from Google Sheets), and Footfall Conversion (walk-ins and orders live from
Odoo, revenue balanced against the finance-reconciled Reporting sheet).
Open access: there is no login page.

Run locally:   streamlit run app.py --server.port 8502
Hosted:        Streamlit Community Cloud, from the main branch (port 8501, its default)
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

nav = st.navigation([
    st.Page("pages/1_Sales.py", title="Sales", icon="💰", default=True),
    st.Page("pages/2_Shop_Attendants.py", title="Shop Attendants", icon="🧑‍💼"),
    st.Page("pages/3_Shop_Performance.py", title="Shop Performance", icon="🏬"),
    st.Page("pages/4_WhatsApp_and_Social_DMs.py", title="WhatsApp & Social DMs", icon="💬"),
    st.Page("pages/5_Footfall_Conversion.py", title="Footfall Conversion", icon="🚶"),
])

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH))
with st.sidebar:
    st.caption("Denri Sales Operations Dashboard")

nav.run()
