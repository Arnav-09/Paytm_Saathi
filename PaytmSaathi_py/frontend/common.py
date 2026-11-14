"""Shared helpers for the Streamlit pages: language, user session, errors, brand bar, footer."""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from ai_engine import accounts
from ai_engine.exceptions import PaytmSaathiError
from ai_engine.llm_provider import provider_status
from ai_engine.policy_extractor import list_policy_ids
from ai_engine.utils import sanitize_collection_name

HINDI_LABEL = "हिन्दी"
LANG_OPTIONS = ["English", HINDI_LABEL]


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
def lang() -> str:
    """'hi' or 'en', from the language toggle (read at call time)."""
    return "hi" if st.session_state.get("lang_choice") == HINDI_LABEL else "en"


def L(en: str, hi: str) -> str:
    return hi if lang() == "hi" else en


def pick(en_list, hi_list):
    """Hindi list when Hindi is selected and available, else English."""
    return hi_list if lang() == "hi" and hi_list else en_list


# ---------------------------------------------------------------------------
# Session / user
# ---------------------------------------------------------------------------
def current_user() -> accounts.PublicUser | None:
    return st.session_state.get("user")


def uid() -> str:
    user = current_user()
    return user.user_id if user else "guest"


def login(user: accounts.PublicUser) -> None:
    st.session_state["user"] = user
    # The language toggle may already be on screen this run; apply on the next run.
    st.session_state["_pending_lang"] = HINDI_LABEL if user.preferred_language == "hi" else "English"


def apply_pending_state() -> None:
    """Call at the very top of the script, before any widget is created."""
    if "_pending_lang" in st.session_state:
        st.session_state["lang_choice"] = st.session_state.pop("_pending_lang")
    st.session_state.setdefault("lang_choice", "English")


def logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------
def run(fn, *args, **kwargs):
    """Call the engine; show a friendly error instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except accounts.AccountError as exc:
        st.error(exc.message_hi if lang() == "hi" else exc.message)
    except PaytmSaathiError as exc:
        st.error(exc.message)
    except Exception as exc:  # unexpected: still no raw traceback for the user
        st.error(L(f"Something went wrong: {exc}", f"कुछ गड़बड़ हो गई: {exc}"))
    return None


@st.cache_data(ttl=60, show_spinner=False)
def ai_status() -> dict:
    return provider_status()


def demo_mode() -> bool:
    return not ai_status()["connected"]


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:100] or "upload"


# Policies are stored as "<user id>_<name>" so users never see each other's policies.
def policy_key(display_name: str) -> str:
    return sanitize_collection_name(f"{uid()}_{display_name}")


def policy_label(policy_id: str) -> str:
    prefix = f"{uid().lower()}_"
    return policy_id[len(prefix):] if policy_id.startswith(prefix) else policy_id


def my_policies() -> list[str]:
    existing = set(list_policy_ids())
    return [p for p in accounts.user_policy_ids(uid()) if p in existing]


def current_policy() -> str | None:
    pid = st.session_state.get("policy_id")
    return pid if pid in my_policies() else None


def need_policy() -> bool:
    """Show a friendly empty state and return False when no policy is selected yet."""
    from frontend.theme import empty_state

    if current_policy():
        return True
    empty_state("📄", L("No policy yet", "अभी कोई पॉलिसी नहीं"),
                L("Upload your health policy in the 'My policy' tab to unlock this.",
                  "इसे इस्तेमाल करने के लिए 'मेरी पॉलिसी' टैब में अपनी हेल्थ पॉलिसी अपलोड करें।"))
    return False


# ---------------------------------------------------------------------------
# Brand bar and footer
# ---------------------------------------------------------------------------
def brand_bar() -> None:
    """Logo, AI status, language toggle and the user menu (works with the sidebar collapsed)."""
    from frontend.theme import esc

    user = current_user()
    status = ai_status()
    c1, c2, c3, c4 = st.columns([5, 3, 2.4, 1.6], vertical_alignment="center")
    c1.markdown('<div class="ps-brand"><span class="ps-logo">S</span>Paytm <b>Saathi</b></div>'
                f'<div class="ps-tag">{esc(L("Your guide for insurance, loans and payments", "बीमा, लोन और पेमेंट में आपका साथी"))}</div>',
                unsafe_allow_html=True)
    if status["connected"]:
        chip = f'<span class="ps-chip"><span class="ps-dot" style="background:#12A26B"></span>AI: {esc(status["label"])} · {esc(status["model"])}</span>'
    else:
        tip = status.get("problem") or L("Add a free Groq key in .env", ".env में मुफ़्त Groq कुंजी जोड़ें")
        chip = (f'<span class="ps-chip" title="{esc(tip)}"><span class="ps-dot" style="background:#D98A00"></span>'
                f'{esc(L("Demo mode (no AI key)", "डेमो मोड (एआई कुंजी नहीं)"))}</span>')
    c2.markdown(chip, unsafe_allow_html=True)
    c3.radio("Language", LANG_OPTIONS, key="lang_choice", horizontal=True, label_visibility="collapsed",
             on_change=_save_language)
    if user:
        with c4.popover(f"👤 {user.first_name}", width="stretch"):
            st.markdown(f"**{user.full_name}**  \n📱 {user.masked_mobile}" + (f"  \n✉️ {user.email}" if user.email else ""))
            st.caption(L("KYC can be completed later from Payments & KYC.",
                         "केवाईसी बाद में 'पेमेंट और केवाईसी' से पूरा कर सकते हैं।"))
            if st.button(L("Log out", "लॉग आउट"), key="logout_btn", width="stretch"):
                logout()
                st.rerun()


def _save_language() -> None:
    user = current_user()
    if user:
        accounts.update_language(user.user_id, lang())


def sidebar() -> None:
    from frontend.theme import esc

    user = current_user()
    with st.sidebar:
        st.markdown('<div class="ps-brand" style="color:#fff"><span class="ps-logo">S</span>Paytm <b>Saathi</b></div>',
                    unsafe_allow_html=True)
        st.write("")
        if user:
            st.markdown(
                f'<div class="ps-side-card"><b>{esc(L("Namaste", "नमस्ते"))}, {esc(user.first_name)} 👋</b><br>'
                f'<span style="font-size:.85rem;opacity:.85">📱 {esc(user.masked_mobile)}</span><br>'
                f'<span style="font-size:.85rem;opacity:.85">📄 {len(my_policies())} {esc(L("policies", "पॉलिसी"))}</span></div>',
                unsafe_allow_html=True)
        st.markdown(
            f'<div class="ps-side-card"><b>{esc(L("Need help fast?", "तुरंत मदद चाहिए?"))}</b><br>'
            f'<span style="font-size:.85rem">🚨 {esc(L("Cyber fraud", "साइबर फ्रॉड"))}: <b>1930</b><br>'
            f'🏦 RBI CMS: cms.rbi.org.in<br>🛡️ IRDAI: bimabharosa.irdai.gov.in</span></div>',
            unsafe_allow_html=True)
        st.caption(L("Saathi never asks for your OTP, PIN or password.",
                     "साथी कभी आपका ओटीपी, पिन या पासवर्ड नहीं माँगता।"))


def footer() -> None:
    from frontend.theme import esc

    st.markdown(f"""
<div class="ps-footer">
  <div class="cols">
    <div><b>{esc(L("About Saathi", "साथी के बारे में"))}</b><br>{esc(L(
        "Saathi explains your policy, loan and payment options in plain language. It does not sell insurance or loans and does not settle claims; insurers and lenders take the final decision.",
        "साथी आपकी पॉलिसी, लोन और पेमेंट विकल्प आसान भाषा में समझाता है। यह बीमा या लोन नहीं बेचता और क्लेम तय नहीं करता; अंतिम फ़ैसला बीमा कंपनी और लेंडर का होता है।"))}</div>
    <div><b>{esc(L("Grievance help", "शिकायत सहायता"))}</b><br>
      {esc(L("Insurance", "बीमा"))}: Bima Bharosa (IRDAI) · bimabharosa.irdai.gov.in<br>
      {esc(L("Banking & payments", "बैंकिंग और पेमेंट"))}: RBI CMS · cms.rbi.org.in<br>
      {esc(L("Cyber fraud", "साइबर फ्रॉड"))}: 1930 · cybercrime.gov.in</div>
    <div><b>{esc(L("Your data", "आपका डेटा"))}</b><br>{esc(L(
        "Documents are processed on this server for your account only. Identity numbers are shown masked and never stored by the KYC check.",
        "दस्तावेज़ केवल आपके खाते के लिए इसी सर्वर पर प्रोसेस होते हैं। पहचान नंबर छिपे हुए दिखते हैं और केवाईसी जाँच में सेव नहीं होते।"))}</div>
  </div>
  <div class="warn">⚠️ {esc(L(
      "Beware of fraud: Saathi, insurers, banks and IRDAI never ask for OTP, PIN or payment to 'release' a claim, refund or loan.",
      "धोखाधड़ी से सावधान: साथी, बीमा कंपनियाँ, बैंक और IRDAI कभी क्लेम, रिफंड या लोन 'छुड़ाने' के लिए ओटीपी, पिन या पैसे नहीं माँगते।"))}</div>
  {esc(L("All figures are estimates for guidance only and are not financial advice, an offer or a guarantee of approval.",
         "सभी आँकड़े केवल मार्गदर्शन के लिए अनुमान हैं; यह वित्तीय सलाह, ऑफर या मंज़ूरी की गारंटी नहीं है।"))}
</div>""", unsafe_allow_html=True)
