"""Payments & KYC journey: payment problems (guidance, complaint draft, escalation dates,
my tickets), scam check with highlighted phrases, KYC readiness stepper."""

from __future__ import annotations

from datetime import date

import streamlit as st

from ai_engine import fintech
from frontend.common import L, current_user, lang, pick, run, uid
from frontend.theme import alert, highlight, note, page_header, pill, steps


def _timeline(ticket: fintech.DisputeTicket) -> None:
    for s in fintech.escalation_timeline(ticket):
        icon = "✅" if s.reached else "🗓️"
        st.markdown(f"{icon} **{s.due_on}** ({L('day', 'दिन')} {s.day}) - {L(s.text, s.text_hi)}")


def _ticket_detail(ticket: fintech.DisputeTicket, key: str) -> None:
    g = ticket.guidance
    if g.urgent:
        alert(L("This looks like fraud. Act now: call 1930, block your UPI/cards and tell your bank today.",
                "यह धोखाधड़ी लगती है। तुरंत कार्रवाई करें: 1930 पर कॉल करें, यूपीआई/कार्ड ब्लॉक करें और आज ही बैंक को बताएँ।"))
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"**{L('What to do now', 'अभी क्या करें')}**")
        for i, text in enumerate(pick(g.steps, g.steps_hi), 1):
            st.markdown(f"{i}. {text}")
        st.markdown(f"**{L('Keep ready', 'तैयार रखें')}:** " + ", ".join(g.evidence_to_collect))
    with right:
        st.markdown(f"**{L('Your escalation dates', 'आपकी कार्रवाई की तारीखें')}**")
        _timeline(ticket)
    _evidence(ticket, key)
    st.markdown(f"**{L('✉️ Complaint ready to send', '✉️ भेजने के लिए तैयार शिकायत')}**")
    who = st.radio(L("Send to", "किसे भेजें"), ["app", "bank", "ombudsman"], horizontal=True, key=f"who_{key}",
                   format_func=lambda r: {"app": L("App support", "ऐप सपोर्ट"), "bank": L("Bank nodal officer", "बैंक नोडल अधिकारी"),
                                          "ombudsman": L("RBI Ombudsman", "आरबीआई लोकपाल")}[r])
    user = current_user()
    letter = fintech.draft_complaint(ticket, user.full_name if user else "", who, user.masked_mobile if user else None)
    st.code(letter, language=None, wrap_lines=True)
    d1, d2 = st.columns(2)
    d1.download_button(L("⬇️ Download complaint", "⬇️ शिकायत डाउनलोड करें"), letter, file_name=f"{ticket.ticket_id}_{who}.txt",
                       mime="text/plain", key=f"dl_{key}", width="stretch")
    pack = fintech.build_evidence_pack(ticket, user.full_name if user else "", who, user.masked_mobile if user else None)
    d2.download_button(L(f"📦 Evidence pack ({len(ticket.attachments)} files, ZIP)", f"📦 सबूत पैक ({len(ticket.attachments)} फ़ाइलें, ZIP)"),
                       pack, file_name=f"{ticket.ticket_id}_{who}_evidence.zip", mime="application/zip", key=f"zip_{key}", width="stretch")
    st.caption(" · ".join(g.helplines))


def _evidence(ticket: fintech.DisputeTicket, key: str) -> None:
    """Attach screenshots / SMS text; they are included in the complaint and the evidence pack."""
    st.markdown(f"**{L('📎 Evidence', '📎 सबूत')}**")
    user = current_user()
    name = user.full_name if user else ""
    if not ticket.transaction_id:
        st.warning(L("Add the transaction ID / UTR to this ticket first. Every SMS and screenshot is checked against it.",
                     "पहले इस टिकट में लेन-देन आईडी / यूटीआर जोड़ें। हर एसएमएस और स्क्रीनशॉट की जाँच उसी से होती है।"))
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        txn = c1.text_input(L("Transaction ID / UTR", "लेन-देन आईडी / यूटीआर"), key=f"settxn_{key}")
        if c2.button(L("Save", "सेव करें"), key=f"settxn_btn_{key}", disabled=not txn.strip()):
            if run(fintech.set_transaction_id, ticket.ticket_id, uid(), txn):
                st.rerun()
        return
    st.caption(L(f"Evidence is accepted only if it belongs to this payment. An SMS must show the date and the transaction ID "
                 f"{ticket.transaction_id}. A screenshot or PDF must show the transaction ID {ticket.transaction_id} and your name "
                 f"({name}). OTPs and PINs are hidden automatically.",
                 f"सबूत तभी स्वीकार होगा जब वह इसी पेमेंट का हो। एसएमएस में तारीख और लेन-देन आईडी {ticket.transaction_id} होनी चाहिए। "
                 f"स्क्रीनशॉट या PDF में लेन-देन आईडी {ticket.transaction_id} और आपका नाम ({name}) दिखना चाहिए। ओटीपी और पिन अपने-आप छिप जाते हैं।"))
    for att in ticket.attachments:
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        icon = {"screenshot": "🖼️", "document": "📄", "sms": "💬"}[att.kind]
        c1.markdown(f"{icon} **{att.filename}** · {att.added_at[:10]}" + (f" · {att.size // 1024} KB" if att.kind != "sms" else ""))
        if att.verified:
            c1.markdown(" ".join(f'<span class="ps-pill ps-ok">✓ {L(c, c)}</span>' for c in att.checks), unsafe_allow_html=True)
        if att.text:
            c1.caption((att.text[:160] + "…") if len(att.text) > 160 else att.text)
        if c2.button("🗑️", key=f"rm_{key}_{att.attachment_id}", help=L("Remove", "हटाएँ")):
            run(fintech.remove_attachment, ticket.ticket_id, uid(), att.attachment_id)
            st.rerun()
    hints = {}
    for att in ticket.attachments:
        hints.update(att.hints)
    missing = [k for k, have in (("utr", ticket.transaction_id), ("amount", ticket.amount), ("date", ticket.transaction_date)) if not have and hints.get(k)]
    if missing:
        found = ", ".join(f"{k.upper() if k == 'utr' else k.title()}: {hints[k]}" for k in missing)
        note(L(f"Found in your evidence - {found}", f"आपके सबूत में मिला - {found}"))
        if st.button(L("Use these details in my ticket", "ये विवरण मेरे टिकट में इस्तेमाल करें"), key=f"hint_{key}"):
            run(fintech.apply_hints_to_ticket, ticket.ticket_id, uid())
            st.rerun()
    if len(ticket.attachments) < fintech.MAX_ATTACHMENTS:
        up = st.file_uploader(L("Screenshot or PDF (PNG, JPG, PDF · up to 5 MB)", "स्क्रीनशॉट या PDF (PNG, JPG, PDF · 5 MB तक)"),
                              type=["png", "jpg", "jpeg", "pdf"], key=f"up_{key}_{len(ticket.attachments)}")
        if up is not None and st.button(L("Attach file", "फ़ाइल जोड़ें"), key=f"att_{key}"):
            with st.spinner(L("Reading and checking the file…", "फ़ाइल पढ़ी और जाँची जा रही है…")):
                added = run(fintech.add_attachment, ticket.ticket_id, uid(), up.name, up.getvalue(), None, name)
            if added:
                st.rerun()
        with st.popover(L("💬 Paste SMS text instead", "💬 इसके बजाय एसएमएस टेक्स्ट पेस्ट करें")):
            sms = st.text_area(L("SMS text", "एसएमएस टेक्स्ट"), key=f"sms_{key}_{len(ticket.attachments)}", height=100)
            if st.button(L("Add SMS", "एसएमएस जोड़ें"), key=f"addsms_{key}", disabled=len(sms.strip()) < 5):
                if run(fintech.add_sms_text, ticket.ticket_id, uid(), sms):
                    st.rerun()


def tab_dispute():
    with st.form("dispute_form"):
        desc = st.text_area(L("What happened?", "क्या हुआ?"), height=100,
                            placeholder=L("e.g. ₹2,000 was deducted but the payment to the shop failed", "जैसे: ₹2,000 कट गए पर दुकान को पेमेंट फेल हो गया"))
        a, b, c = st.columns(3)
        amount = a.number_input(L("Amount (₹)", "राशि (₹)"), 0.0, 10000000.0, 0.0)
        txn = b.text_input(L("Transaction ID / UTR *", "लेन-देन आईडी / यूटीआर *"),
                           help=L("Required. Find it in your payment app or debit SMS (UTR / UPI Ref No / Txn ID).",
                                  "ज़रूरी। यह आपके पेमेंट ऐप या डेबिट एसएमएस में मिलेगा (UTR / UPI Ref No / Txn ID)।"))
        when = c.date_input(L("Date of payment", "पेमेंट की तारीख"), value=date.today(), max_value=date.today(), format="DD/MM/YYYY")
        go = st.form_submit_button(L("Get help and create ticket", "मदद लें और टिकट बनाएँ"), type="primary")
    if go:
        if not desc.strip():
            st.error(L("Tell us what happened.", "बताइए क्या हुआ।"))
        elif not txn.strip():
            st.error(L("Enter the transaction ID / UTR - your evidence will be checked against it.",
                       "लेन-देन आईडी / यूटीआर दर्ज करें - आपके सबूत की जाँच उसी से होगी।"))
        else:
            ticket = run(fintech.create_dispute_ticket, desc, amount or None, txn or None, None, None, uid(), when.isoformat())
            if ticket:
                st.session_state["last_ticket"] = ticket.ticket_id
    tickets = fintech.list_dispute_tickets(uid())
    last = next((t for t in tickets if t.ticket_id == st.session_state.get("last_ticket")), None)
    if last:
        g = last.guidance
        st.markdown(f"### {L(g.title, g.title_hi)}")
        st.caption(f"{L('Ticket', 'टिकट')} **{last.ticket_id}** · {L(g.expected_timeline, g.expected_timeline_hi)}")
        _ticket_detail(last, "new")
    others = [t for t in tickets if t is not last]
    if others:
        st.markdown(f"#### {L('My tickets', 'मेरे टिकट')}")
        for t in others:
            with st.expander(f"{t.ticket_id} · {L(t.guidance.title, t.guidance.title_hi)} · {t.created_at[:10]}"):
                _ticket_detail(t, t.ticket_id)


def tab_scam():
    text = st.text_area(L("Paste the SMS, WhatsApp message or what the caller said", "एसएमएस, व्हाट्सऐप संदेश या कॉलर ने जो कहा वह पेस्ट करें"),
                        height=130, key="scam_txt",
                        placeholder=L("e.g. Dear customer, your KYC expires today. Share OTP to avoid account block…",
                                      "जैसे: प्रिय ग्राहक, आपका केवाईसी आज खत्म हो रहा है। खाता बंद होने से बचाने के लिए ओटीपी बताएँ…"))
    sender = st.text_input(L("Who sent it? (optional: SMS name like VM-HDFCBK, or the phone number)",
                             "किसने भेजा? (वैकल्पिक: VM-HDFCBK जैसा एसएमएस नाम, या फ़ोन नंबर)"), key="scam_sender")
    if not st.button(L("Check message", "संदेश जाँचें"), type="primary", disabled=not text.strip(), key="scam_go"):
        st.caption(L("Nothing you paste here is stored. We check the wording, links, phone numbers and sender.",
                     "यहाँ पेस्ट किया गया कुछ भी सेव नहीं होता। हम शब्दों, लिंक, फ़ोन नंबर और भेजने वाले की जाँच करते हैं।"))
        return
    res = run(fintech.check_scam_message, text, sender or None)
    if not res:
        return
    kind, label, icon = {"high": ("bad", L("High risk - likely a scam", "ज़्यादा जोखिम - ठगी की संभावना"), "🛑"),
                         "medium": ("warn", L("Be careful", "सावधान रहें"), "⚠️"),
                         "low": ("ok", L("Low risk", "कम जोखिम"), "✅")}[res.risk]
    st.markdown(f"### {icon} {pill(label, kind)}", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(f"**{L('Risky parts of the message', 'संदेश के जोखिम भरे हिस्से')}**")
        st.markdown(highlight(text, res.spans), unsafe_allow_html=True)
        for reason in pick(res.reasons, res.reasons_hi):
            st.markdown(f"- {reason}")
    with right:
        st.markdown(f"**{L('Do this now', 'अभी यह करें')}**")
        for advice in pick(res.advice, res.advice_hi):
            note(advice)
    _scam_details(res)
    st.caption(res.disclaimer)


_VERDICT = {"official": ("ok", "✅", "Official domain", "आधिकारिक डोमेन"), "dangerous": ("bad", "🛑", "Dangerous", "खतरनाक"),
            "suspicious": ("warn", "⚠️", "Suspicious", "संदिग्ध"), "unknown": ("info", "❔", "Unknown", "अज्ञात")}


def _scam_details(res) -> None:
    if res.claimed_brands:
        st.markdown(f"**{L('The message claims to be from', 'संदेश खुद को किसका बताता है')}:** " + ", ".join(res.claimed_brands))
    if res.links:
        st.markdown(f"**{L('🔗 Links found', '🔗 मिले लिंक')}**")
        for link in res.links:
            kind, icon, en, hi = _VERDICT[link.verdict]
            st.markdown(f"{icon} `{link.domain or link.url}` · {pill(L(en, hi), kind)}", unsafe_allow_html=True)
            for reason in pick(link.reasons[:3], link.reasons_hi[:3]):
                st.caption(f"• {reason}")
    if res.phones:
        st.markdown(f"**{L('📞 Phone numbers found', '📞 मिले फ़ोन नंबर')}**")
        for ph in res.phones:
            st.markdown(("⚠️ " if ph.verdict == "suspicious" else "ℹ️ ") + f"`{ph.number}` - {L(ph.note, ph.note_hi)}")
    if res.sender:
        icon = {"consistent": "✅", "mismatch": "⚠️", "suspicious": "⚠️", "unknown": "❔"}[res.sender.verdict]
        st.markdown(f"**{L('📨 Sender check', '📨 भेजने वाले की जाँच')}**  \n{icon} `{res.sender.sender}` - {L(res.sender.note, res.sender.note_hi)}")


def tab_kyc():
    st.caption(L("Your account works without KYC. Full KYC unlocks higher limits - check you're ready before you start.",
                 "आपका खाता बिना केवाईसी के चलता है। पूर्ण केवाईसी से ज़्यादा सीमा मिलती है - शुरू करने से पहले तैयारी जाँचें।"))
    level = st.radio(L("KYC level", "केवाईसी स्तर"), ["minimum", "full"], horizontal=True, key="kyc_lvl", index=1,
                     format_func=lambda k: L(fintech.KYC_LEVELS[k]["title"], fintech.KYC_LEVELS[k]["title_hi"]))
    note(L(fintech.KYC_LEVELS[level]["limits"], fintech.KYC_LEVELS[level]["limits_hi"]))
    user = current_user()
    a, b, c = st.columns(3)
    mobile = a.text_input(L("Mobile number", "मोबाइल नंबर"), value=user.mobile if user else "", key="kyc_m")
    aadhaar = b.text_input(L("Aadhaar number", "आधार नंबर"), type="password", key="kyc_a")
    pan = c.text_input(L("PAN", "पैन"), key="kyc_p")
    names = st.checkbox(L("My name is spelled the same on Aadhaar and PAN", "आधार और पैन में मेरे नाम की स्पेलिंग एक जैसी है"), value=True, key="kyc_n")
    st.caption(L("Checked only for format on this server and shown masked. Nothing is saved.",
                 "केवल फॉर्मेट जाँचा जाता है और छिपा हुआ दिखता है। कुछ भी सेव नहीं होता।"))
    if not st.button(L("Check readiness", "तैयारी जाँचें"), type="primary", key="kyc_go"):
        return
    res = run(fintech.assess_kyc, level, mobile, aadhaar, pan, names)
    if not res:
        return
    ok = sum(1 for chk in res.checks if chk.valid)
    total = 3 if level == "full" else 2
    steps([L("Documents ready", "दस्तावेज़ तैयार"), L("OTP verification", "ओटीपी सत्यापन")]
          + ([L("Video / in-person check", "वीडियो / सामने से जाँच")] if level == "full" else []),
          1 if res.ready else 0)
    st.progress(ok / total, text=L(f"{ok} of {total} details look correct", f"{total} में से {ok} विवरण सही हैं"))
    for chk in res.checks:
        st.markdown(("✅ " if chk.valid else "❌ ") + f"`{chk.masked_value}` · " + L(chk.message, chk.message_hi))
    for text in pick(res.missing, res.missing_hi):
        st.markdown(f"- {text}")
    st.markdown(f"**{L('Next steps', 'अगले कदम')}**")
    for i, text in enumerate(pick(res.next_steps, res.next_steps_hi), 1):
        st.markdown(f"{i}. {text}")
    st.caption(L(res.note, res.note_hi))


def render():
    page_header(L("Payments & KYC", "पेमेंट और केवाईसी"),
                L("Get stuck money back, stay ahead of scams, and get KYC-ready.",
                  "अटका पैसा वापस पाएँ, ठगी से बचें और केवाईसी के लिए तैयार रहें।"))
    tabs = st.tabs([L("💸 Payment problem", "💸 पेमेंट समस्या"), L("🛡️ Scam check", "🛡️ ठगी जाँच"), L("🪪 KYC readiness", "🪪 केवाईसी तैयारी")])
    for tab, fn in zip(tabs, (tab_dispute, tab_scam, tab_kyc)):
        with tab:
            fn()
