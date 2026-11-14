"""Home: greeting, journeys, what needs attention, quick tips."""

from __future__ import annotations

import streamlit as st

from ai_engine import fintech
from frontend.common import L, current_user, demo_mode, my_policies, uid
from frontend.theme import card, esc, stat


def _go(url_path: str, label: str, key: str) -> None:
    page = st.session_state.get("_pages", {}).get(url_path)
    if page is not None:
        st.page_link(page, label=label, icon="➡️")


def render() -> None:
    user = current_user()
    st.markdown(
        '<div class="ps-hero">'
        f'<div class="ps-hello">{esc(L("Namaste", "नमस्ते"))}, {esc(user.first_name if user else "")} 👋</div>'
        f'<h1>{esc(L("Insurance, loans and payments - simpler, faster, more human.", "बीमा, लोन और पेमेंट - आसान, तेज़ और अपनापन भरा।"))}</h1>'
        f'<p>{esc(L("Ask in plain English or Hindi, understand your documents, and finish important money tasks with confidence.", "सरल हिंदी या अंग्रेज़ी में पूछें, अपने दस्तावेज़ समझें और पैसों के ज़रूरी काम भरोसे के साथ पूरे करें।"))}</p>'
        "</div>", unsafe_allow_html=True)

    chat_page = st.session_state.get("_pages", {}).get("chat")
    if chat_page is not None:
        st.page_link(chat_page, label=L("Ask Saathi anything - insurance, loans, payments or general questions", "साथी से कुछ भी पूछिए - बीमा, लोन, पेमेंट या कोई भी सवाल"), icon="💬")
    tickets = fintech.list_dispute_tickets(uid())
    open_tickets = [t for t in tickets if t.status == "open"]
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(stat(L("Policies uploaded", "अपलोड की गई पॉलिसी"), str(len(my_policies()))), unsafe_allow_html=True)
    s2.markdown(stat(L("Open payment issues", "खुली पेमेंट शिकायतें"), str(len(open_tickets))), unsafe_allow_html=True)
    s3.markdown(stat(L("Languages", "भाषाएँ"), "English · हिन्दी"), unsafe_allow_html=True)
    s4.markdown(stat(L("AI assistant", "एआई सहायक"), L("Demo mode", "डेमो मोड") if demo_mode() else L("Connected", "जुड़ा हुआ")),
                unsafe_allow_html=True)
    st.write("")

    st.markdown(f"### {L('What would you like to do?', 'आप क्या करना चाहेंगे?')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(card("🛡️", L("Health insurance", "हेल्थ बीमा"),
                         L("Know what your policy really covers - before you need it.", "ज़रूरत से पहले जानें कि पॉलिसी में असल में क्या कवर है।"),
                         [L("Coverage at a glance", "एक नज़र में कवरेज"), L("Claim Pre-Flight payout estimate", "क्लेम प्री-फ्लाइट भुगतान अनुमान"),
                          L("Waiting-period countdown", "प्रतीक्षा अवधि काउंटडाउन")]), unsafe_allow_html=True)
        _go("insurance", L("Open insurance", "बीमा खोलें"), "go_ins")
    with c2:
        st.markdown(card("💳", L("Loans", "लोन"),
                         L("Borrow with your eyes open: EMI, eligibility and true cost.", "समझदारी से उधार लें: ईएमआई, पात्रता और असली लागत।"),
                         [L("Prepayment planner", "प्रीपेमेंट प्लानर"), L("What would raise my eligibility?", "मेरी पात्रता कैसे बढ़े?"),
                          L("Compare offers including fees", "शुल्क सहित ऑफर की तुलना")]), unsafe_allow_html=True)
        _go("loans", L("Open loans", "लोन खोलें"), "go_loan")
    with c3:
        st.markdown(card("📲", L("Payments & KYC", "पेमेंट और केवाईसी"),
                         L("Get stuck money back and stay safe from scams.", "अटका पैसा वापस पाएँ और ठगी से बचें।"),
                         [L("Complaint letter drafted for you", "आपके लिए शिकायत पत्र तैयार"), L("Escalation dates you can act on", "कार्रवाई की तारीखें"),
                          L("Scam message check", "ठगी संदेश जाँच")]), unsafe_allow_html=True)
        _go("payments", L("Open payments", "पेमेंट खोलें"), "go_pay")

    if open_tickets:
        st.write("")
        st.markdown(f"### {L('Needs your attention', 'आपका ध्यान चाहिए')}")
        for t in open_tickets[:3]:
            steps = fintech.escalation_timeline(t)
            nxt = next((s for s in steps if not s.reached), None)
            title = L(t.guidance.title, t.guidance.title_hi)
            when = f" · {L('next step on', 'अगला कदम')} {nxt.due_on}: {L(nxt.text, nxt.text_hi)}" if nxt else ""
            st.markdown(f"- **{t.ticket_id}** · {title}{when}")

    st.write("")
    st.markdown(f"### {L('Good to know', 'जानने लायक')}")
    t1, t2, t3 = st.columns(3)
    t1.info(L("Choosing a hospital room above your room-rent limit can cut the payout on doctor and surgery charges too, not just the room.",
              "रूम-रेंट सीमा से महँगा कमरा लेने पर सिर्फ कमरा ही नहीं, डॉक्टर और सर्जरी का भुगतान भी कट सकता है।"), icon="💡")
    t2.info(L("A small yearly prepayment on a long home loan can save years of EMIs. Try the prepayment planner.",
              "लंबे होम लोन पर छोटा सालाना प्रीपेमेंट कई साल की ईएमआई बचा सकता है। प्रीपेमेंट प्लानर आज़माएँ।"), icon="💡")
    t3.info(L("You never need a UPI PIN to receive money. A 'collect request' asking for your PIN is a scam.",
              "पैसे पाने के लिए कभी यूपीआई पिन नहीं लगता। पिन माँगने वाली 'कलेक्ट रिक्वेस्ट' ठगी है।"), icon="🛑")
