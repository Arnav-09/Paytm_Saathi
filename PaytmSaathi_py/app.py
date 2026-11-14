"""Paytm Saathi - Streamlit app.

Run from the project root:   streamlit run app.py

The UI calls the AI engine in-process. The FastAPI service (backend/main.py) exposes the
same functions over HTTP for other clients:   uvicorn backend.main:app --reload
"""

from __future__ import annotations

import streamlit as st

from frontend.theme import inject

st.set_page_config(page_title="Paytm Saathi", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")
inject()

# Engine imports come after page config so the page paints quickly.
from frontend import auth_ui, fintech_ui, home_ui, insurance_ui, lending_ui  # noqa: E402
from frontend.common import L, apply_pending_state, brand_bar, current_user, footer, sidebar  # noqa: E402

apply_pending_state()

if current_user() is None:
    auth_ui.render()
    footer()
    st.stop()

pages = [
    st.Page(home_ui.render, title=L("Home", "होम"), icon="🏠", url_path="home", default=True),
    st.Page(insurance_ui.render, title=L("Insurance", "बीमा"), icon="🛡️", url_path="insurance"),
    st.Page(lending_ui.render, title=L("Loans", "लोन"), icon="💳", url_path="loans"),
    st.Page(fintech_ui.render, title=L("Payments & KYC", "पेमेंट और केवाईसी"), icon="📲", url_path="payments"),
]
nav = st.navigation(pages, position="top")
st.session_state["_pages"] = {p.url_path: p for p in pages}
brand_bar()
sidebar()
nav.run()
footer()
