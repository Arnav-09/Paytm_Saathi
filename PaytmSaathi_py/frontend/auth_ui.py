"""Login and registration (basic details only; KYC comes later)."""

from __future__ import annotations

import streamlit as st

from ai_engine import accounts
from frontend.common import HINDI_LABEL, LANG_OPTIONS, L, login, run
from frontend.theme import esc


def render() -> None:
    top_l, top_r = st.columns([4, 1.4], vertical_alignment="center")
    top_l.markdown('<div class="ps-brand"><span class="ps-logo">S</span>Paytm <b>Saathi</b></div>', unsafe_allow_html=True)
    top_r.radio("Language", LANG_OPTIONS, key="lang_choice", horizontal=True, label_visibility="collapsed")
    st.write("")

    side, main = st.columns([1, 1.15], gap="large")
    with side:
        points = [
            L("Understand your health policy in plain language", "अपनी हेल्थ पॉलिसी आसान भाषा में समझें"),
            L("Know your claim payout before you pay the hospital", "अस्पताल को भुगतान से पहले क्लेम की राशि जानें"),
            L("Check loan eligibility and the real cost of an offer", "लोन की पात्रता और ऑफर की असली लागत जानें"),
            L("Fix failed payments and spot scams", "फेल पेमेंट ठीक करें और ठगी पहचानें"),
        ]
        st.markdown(
            '<div class="ps-auth-side"><span class="ps-logo" style="width:48px;height:48px;font-size:1.4rem">S</span>'
            f'<h2>{esc(L("Money matters, made simple.", "पैसों के काम, अब आसान।"))}</h2>'
            f'<p>{esc(L("One place for insurance, loans and payments - in English or Hindi.", "बीमा, लोन और पेमेंट - एक ही जगह, हिंदी या अंग्रेज़ी में।"))}</p>'
            "<ul>" + "".join(f"<li>{esc(p)}</li>" for p in points) + "</ul>"
            f'<p style="margin-top:1.4rem;font-size:.82rem">🔒 {esc(L("We only need basic details to start. No KYC documents at sign-up.", "शुरू करने के लिए केवल बुनियादी जानकारी चाहिए। साइन-अप पर कोई केवाईसी दस्तावेज़ नहीं।"))}</p>'
            "</div>", unsafe_allow_html=True)

    with main:
        tab_login, tab_register = st.tabs([L("Log in", "लॉग इन"), L("Create account", "खाता बनाएँ")])
        with tab_login:
            _login_form()
        with tab_register:
            _register_form()


def _login_form() -> None:
    with st.form("login_form"):
        st.markdown(f"#### {L('Welcome back', 'वापसी पर स्वागत है')}")
        ident = st.text_input(L("Mobile number or email", "मोबाइल नंबर या ईमेल"), placeholder="98765 43210")
        password = st.text_input(L("Password", "पासवर्ड"), type="password")
        submitted = st.form_submit_button(L("Log in", "लॉग इन"), type="primary", width="stretch")
    if submitted:
        user = run(accounts.authenticate, ident, password)
        if user:
            login(user)
            st.rerun()
    st.caption(L("Forgot password? Password reset by OTP will come with mobile verification.",
                 "पासवर्ड भूल गए? मोबाइल सत्यापन के साथ ओटीपी से पासवर्ड रीसेट जल्द आएगा।"))


def _register_form() -> None:
    with st.form("register_form", clear_on_submit=False):
        st.markdown(f"#### {L('Create your account', 'अपना खाता बनाएँ')}")
        name = st.text_input(L("Full name", "पूरा नाम"))
        c1, c2 = st.columns(2)
        mobile = c1.text_input(L("Mobile number", "मोबाइल नंबर"), placeholder="98765 43210")
        email = c2.text_input(L("Email (optional)", "ईमेल (वैकल्पिक)"))
        c3, c4 = st.columns(2)
        password = c3.text_input(L("Password", "पासवर्ड"), type="password",
                                 help=L("At least 8 characters with a letter and a number.", "कम से कम 8 अक्षर, जिसमें एक अक्षर और एक अंक हो।"))
        confirm = c4.text_input(L("Confirm password", "पासवर्ड दोबारा"), type="password")
        language = st.radio(L("Preferred language", "पसंदीदा भाषा"), LANG_OPTIONS, horizontal=True,
                            index=1 if st.session_state.get("lang_choice") == HINDI_LABEL else 0)
        terms = st.checkbox(L("I agree to the terms of use and privacy policy", "मैं उपयोग की शर्तों और गोपनीयता नीति से सहमत हूँ"))
        submitted = st.form_submit_button(L("Create account", "खाता बनाएँ"), type="primary", width="stretch")
    if submitted:
        user = run(accounts.register_user, name, mobile, password, confirm, email or None,
                   "hi" if language == HINDI_LABEL else "en", terms)
        if user:
            login(user)
            st.success(L(f"Welcome, {user.first_name}! Your account is ready.", f"स्वागत है, {user.first_name}! आपका खाता तैयार है।"))
            st.rerun()
