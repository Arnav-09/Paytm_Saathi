"""Inline chat cards for Money Clarity and Loan Pre-Flight (no separate pages).

* "📎 Add statement or document" and "🏦 Loan details" boxes sit above the chat.
* Results appear as cards inside assistant messages: Money Clarity report, Loan Pre-Flight, rejection decoder.
* Everything uploaded or entered stays in the user's session only (nothing is written to disk).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

from ai_engine import lending, money, statements
from ai_engine.exceptions import PaytmSaathiError
from ai_engine.session_tools import SessionData
from frontend.common import L, pick, run, uid
from ai_engine.utils import inr
from frontend.theme import CYAN, NAVY, alert, esc, note, pill

STATUS_KIND = {"safe": "ok", "tight": "warn", "high_risk": "bad", "ok": "ok", "shortfall": "bad", "unknown": "info"}
CONF_KIND = {"high": "ok", "medium": "warn", "low": "info"}


# ---------------------------------------------------------------------------
# Session state (per user, in memory only)
# ---------------------------------------------------------------------------
def _box() -> dict:
    return st.session_state.setdefault("money_session", {}).setdefault(
        uid(), {"files": {}, "docs": [], "loan": None, "loan_runs": [], "balance": None, "salary": None})


def session_data() -> SessionData:
    box = _box()
    stmts = list(box["files"].values())
    merged = statements.merge_statements(stmts) if stmts else None
    return SessionData(statement=merged, loan_inputs=box["loan"], documents=box["docs"], as_of=date.today(),
                       balance=box["balance"], next_salary=box["salary"])


def session_note() -> str:
    box, parts = _box(), []
    if box["files"]:
        s = session_data().statement
        parts.append(f"Statement uploaded: {', '.join(box['files'])} ({len(s.transactions)} transactions, "
                     f"{s.period_start} to {s.period_end}, type {s.kind}).")
    if box["loan"]:
        i = box["loan"]
        parts.append(f"Loan details saved: {inr(i.loan_amount)} for {i.tenure_months} months at {i.annual_rate_pct}% "
                     f"(income {inr(i.monthly_income)}/month).")
    if box["docs"]:
        parts.append("Documents uploaded: " + ", ".join(name for name, _ in box["docs"]) + ".")
    return " ".join(parts)


def save_loan_inputs(inputs: dict) -> Optional[lending.LoanPreflightResult]:
    """Store a simulation (inputs + outputs) for this session."""
    try:
        inp = lending.LoanPreflightInput(**inputs)
        res = lending.loan_preflight(inp)
    except (PaytmSaathiError, ValueError):
        return None
    box = _box()
    box["loan"] = inp
    box["loan_runs"] = (box["loan_runs"] + [{"inputs": inp.model_dump(), "emi": res.emi, "status": res.status}])[-10:]
    return res


# ---------------------------------------------------------------------------
# Boxes above the chat
# ---------------------------------------------------------------------------
def upload_box(add_turn) -> None:
    box = _box()
    label = L("📎 Add statement or document", "📎 स्टेटमेंट या दस्तावेज़ जोड़ें")
    if box["files"] or box["docs"]:
        label += f" · {len(box['files']) + len(box['docs'])}"
    with st.expander(label):
        st.caption(L("Bank or credit-card statement (CSV/TSV, Excel or PDF, any bank's layout) for Money Clarity, or a loan rejection letter / "
                     "credit report (PDF, image, text). Kept only for this session; card and account numbers are masked.",
                     "मनी क्लैरिटी के लिए बैंक या क्रेडिट-कार्ड स्टेटमेंट (CSV, Excel या PDF), या लोन रिजेक्शन लेटर / क्रेडिट "
                     "रिपोर्ट (PDF, फोटो, टेक्स्ट)। केवल इस सेशन में रहता है; कार्ड और खाता नंबर छिपा दिए जाते हैं।"))
        files = st.file_uploader(L("Choose files", "फ़ाइलें चुनें"), type=["csv", "tsv", "xlsx", "xls", "pdf", "png", "jpg", "jpeg", "txt"],
                                 accept_multiple_files=True, key=f"money_up_{len(box['files'])}_{len(box['docs'])}")
        c1, c2 = st.columns(2)
        bal = c1.number_input(L("Current bank balance (₹, optional)", "अभी बैंक बैलेंस (₹, वैकल्पिक)"), min_value=0.0,
                              value=float(box["balance"] or 0.0), step=500.0, key="money_bal")
        sal = c2.date_input(L("Next salary date (optional)", "अगली सैलरी की तारीख (वैकल्पिक)"), value=box["salary"],
                            min_value=date.today(), key="money_sal", format="DD/MM/YYYY")
        box["balance"], box["salary"] = (bal or None), sal
        if files and st.button(L("Read files", "फ़ाइलें पढ़ें"), type="primary", key="money_read"):
            added_stmt, added_doc = [], []
            for f in files:
                data = f.getvalue()
                try:
                    box["files"][f.name] = statements.parse_statement(f.name, data)
                    added_stmt.append(f.name)
                except PaytmSaathiError as exc:
                    if f.name.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".txt")):
                        lines = run(lending.document_lines, f.name, data)
                        if lines:
                            box["docs"] = [d for d in box["docs"] if d[0] != f.name] + [(f.name, lines)]
                            added_doc.append(f.name)
                    else:
                        st.error(f"{f.name}: {exc.message}")
            if added_stmt:
                add_turn(L(f"I read {', '.join(added_stmt)}. Here is your Money Clarity report - ask me anything about it.",
                           f"मैंने {', '.join(added_stmt)} पढ़ लिया। यह रही आपकी मनी क्लैरिटी रिपोर्ट - इसके बारे में कुछ भी पूछिए।"),
                         ["money"])
            if added_doc:
                add_turn(L(f"I read {', '.join(added_doc)}.", f"मैंने {', '.join(added_doc)} पढ़ लिया।"), ["rejection"])
            st.rerun()
        if box["files"] or box["docs"]:
            for name in list(box["files"]) + [n for n, _ in box["docs"]]:
                a, b = st.columns([5, 1], vertical_alignment="center")
                a.markdown(f"📄 {esc(name)}", unsafe_allow_html=True)
                if b.button("✕", key=f"money_rm_{name}", help=L("Remove", "हटाएँ")):
                    box["files"].pop(name, None)
                    box["docs"] = [d for d in box["docs"] if d[0] != name]
                    st.rerun()


def loan_box(add_turn) -> None:
    box = _box()
    saved = box["loan"]
    label = L("🏦 Loan details", "🏦 लोन विवरण") + (f" · {inr(saved.loan_amount)}" if saved else "")
    with st.expander(label):
        d = saved.model_dump() if saved else {}
        with st.form("loan_box_form"):
            a, b, c = st.columns(3)
            amount = a.number_input(L("Loan amount (₹)", "लोन राशि (₹)"), 10000.0, 50000000.0, float(d.get("loan_amount", 500000)), 10000.0)
            tenure = b.slider(L("Tenure (months)", "अवधि (महीने)"), 6, 360, int(d.get("tenure_months", 36)), 6)
            rate = c.number_input(L("Interest rate quoted (% a year)", "बताई गई ब्याज दर (% सालाना)"), 0.0, 60.0,
                                  float(d.get("annual_rate_pct", 13.5)), 0.25)
            a, b, c = st.columns(3)
            income = a.number_input(L("Monthly income (₹)", "मासिक आय (₹)"), 0.0, 10000000.0, float(d.get("monthly_income", 85000)), 1000.0)
            emis = b.number_input(L("Existing EMIs / month (₹)", "मौजूदा ईएमआई / माह (₹)"), 0.0, 10000000.0, float(d.get("existing_emis", 12500)), 500.0)
            card = c.number_input(L("Card dues / month (₹)", "कार्ड बकाया / माह (₹)"), 0.0, 10000000.0, float(d.get("card_dues_monthly", 5000)), 500.0)
            a, b, c = st.columns(3)
            expenses = a.number_input(L("Other monthly expenses (₹, 0 = skip)", "अन्य मासिक खर्च (₹, 0 = छोड़ें)"), 0.0, 10000000.0,
                                      float(d.get("monthly_expenses") or 40000), 1000.0)
            savings = b.number_input(L("Savings for emergencies (₹)", "आपात बचत (₹)"), 0.0, 100000000.0, float(d.get("savings") or 30000), 1000.0)
            score = c.selectbox(L("Credit score range", "क्रेडिट स्कोर रेंज"), list(lending.SCORE_BANDS),
                                index=list(lending.SCORE_BANDS).index(d.get("credit_score_range", "unknown")))
            a, b = st.columns(2)
            emp = a.selectbox(L("Employment", "रोज़गार"), ["salaried", "self_employed"],
                              index=0 if d.get("employment_type", "salaried") == "salaried" else 1)
            apps = b.number_input(L("Loan/card applications in last 6 months", "पिछले 6 महीने में आवेदन"), 0, 50, int(d.get("recent_applications", 0)))
            go = st.form_submit_button(L("Save & check with Saathi", "सेव करें और साथी से जाँचें"), type="primary")
        if go:
            res = save_loan_inputs(dict(loan_amount=amount, tenure_months=int(tenure), annual_rate_pct=rate, monthly_income=income,
                                        existing_emis=emis, card_dues_monthly=card, monthly_expenses=expenses or None,
                                        savings=savings, credit_score_range=score, employment_type=emp, recent_applications=int(apps)))
            if res is None:
                st.error(L("Please check the numbers (income and amount must be more than zero).",
                           "कृपया संख्याएँ जाँचें (आय और राशि शून्य से ज़्यादा होनी चाहिए)।"))
            else:
                add_turn(L(f"Loan Pre-Flight for {inr(amount)} over {tenure} months at {rate}%: EMI {inr(res.emi)}, "
                           f"status **{lending.STATUS_LABELS[res.status]}**. Ask me 'what if' questions below.",
                           f"{inr(amount)} के लोन ({tenure} महीने, {rate}%) का प्री-फ्लाइट: ईएमआई {inr(res.emi)}, स्थिति "
                           f"**{lending.STATUS_LABELS[res.status]}**। नीचे 'क्या हो अगर' सवाल पूछिए।"), ["loan"])
                st.rerun()


# ---------------------------------------------------------------------------
# Cards (rendered inside assistant messages)
# ---------------------------------------------------------------------------
def snapshot(cards: list[str], artifacts: Optional[dict] = None) -> dict:
    """Freeze card data when the message is created, so old messages keep showing what they showed."""
    data = session_data()
    out: dict = {}
    if "money" in cards and data.statement:
        out["money"] = money.analyse(data.statement, data.as_of, data.balance, data.next_salary).model_dump()
        out["transactions"] = [t.model_dump(include={"txn_id", "date", "merchant", "amount", "type", "category",
                                                     "recurring", "duplicate", "fee", "source"}) for t in data.statement.transactions]
    if "loan" in cards:
        inputs = (artifacts or {}).get("loan_inputs")
        if inputs:  # numbers the model used from chat -> save as this session's simulation
            save_loan_inputs(inputs)
            data = session_data()
        if data.loan_inputs:
            out["loan"] = lending.loan_preflight(data.loan_inputs).model_dump()
    if "rejection" in cards and data.documents:
        out["rejection"] = [lending.decode_rejection(lines, name).model_dump() for name, lines in data.documents]
    return out


def _evidence_lines(evidence: list[dict]) -> None:
    for e in evidence[:4]:
        st.caption(f"[{e['ref']}] {e['date']} · {e['merchant']} · {inr(e['amount'], 2)} · {e['source']}")


def render_cards(turn: dict, key: str, ask) -> None:
    data = turn.get("card_data") or {}
    if "money" in data:
        _money_card(data["money"], data.get("transactions", []), key, ask)
    if "loan" in data:
        _loan_card(data["loan"], key, ask)
    for decoded in data.get("rejection", []):
        _rejection_card(decoded)


def _money_card(r: dict, txns: list[dict], key: str, ask) -> None:
    with st.container(border=True):
        st.markdown(f"**{L('💰 Money Clarity report', '💰 मनी क्लैरिटी रिपोर्ट')}** · {esc(r['source'])} · "
                    f"{r['period_start']} → {r['period_end']}", unsafe_allow_html=True)
        cf = r["cash_flow"]
        m = st.columns(4)
        m[0].metric(L("Money out", "खर्च"), inr(r["total_out"]))
        m[1].metric(L("Money in", "आमदनी"), inr(r["total_in"]))
        m[2].metric(L("Possible savings / month", "संभावित बचत / माह"), inr(r["potential_monthly_savings"]))
        m[3].markdown(f"{L('Cash flow', 'नकदी')}<br>{pill({'ok': L('OK', 'ठीक'), 'tight': L('Tight', 'तंग'), 'shortfall': L('Shortfall', 'कमी'), 'unknown': L('Add balance', 'बैलेंस जोड़ें')}[cf['status']], STATUS_KIND[cf['status']])}",
                      unsafe_allow_html=True)
        if cf["status"] in ("shortfall", "tight"):
            alert(cf["message"])
        elif cf["message"]:
            note(cf["message"])
        tabs = st.tabs([L("Alerts", "अलर्ट"), L("Upcoming dues", "आने वाले भुगतान"), L("Recurring", "नियमित"),
                        L("Spending", "खर्च"), L("Action plan", "कार्य योजना"), L("Transactions", "लेन-देन")])
        with tabs[0]:
            if not r["insights"]:
                st.caption(L("No duplicates, fees or unusual charges found.", "कोई दोहरा चार्ज, शुल्क या असामान्य खर्च नहीं मिला।"))
            for i in r["insights"]:
                st.markdown(f"**{esc(i['title'])}** {pill(i['confidence'], CONF_KIND[i['confidence']])}" +
                            (f" · {inr(i['amount'], 2)}" if i.get("amount") else ""), unsafe_allow_html=True)
                st.caption(i["detail"])
                _evidence_lines(i["evidence"])
        with tabs[1]:
            if r["upcoming"]:
                st.dataframe(pd.DataFrame([{L("Due", "तारीख"): o["due_date"], L("What", "क्या"): o["name"], L("Type", "प्रकार"): o["kind"],
                                            L("Amount", "राशि"): inr(o["amount"], 2), L("Confidence", "भरोसा"): o["confidence"],
                                            L("Why we think so", "आधार"): o["basis"]} for o in r["upcoming"]]),
                             hide_index=True, width="stretch")
            else:
                st.caption(L("No bills or EMIs expected in the next 45 days from this statement.", "इस स्टेटमेंट से अगले 45 दिनों में कोई बिल या ईएमआई नहीं दिखती।"))
        with tabs[2]:
            if r["recurring"]:
                st.dataframe(pd.DataFrame([{L("Merchant", "व्यापारी"): x["merchant"], L("Category", "श्रेणी"): x["category"],
                                            L("Amount", "राशि"): inr(x["average_amount"], 2), L("How often", "कितनी बार"): x["frequency"],
                                            L("Next", "अगला"): x["next_expected"], L("Confidence", "भरोसा"): x["confidence"],
                                            L("Source", "स्रोत"): x["evidence"][-1]["source"] if x["evidence"] else ""}
                                           for x in r["recurring"]]), hide_index=True, width="stretch")
        with tabs[3]:
            if r["categories"]:
                cats = pd.DataFrame({L("Spent (₹)", "खर्च (₹)"): [c["total"] for c in r["categories"][:8]]},
                                    index=[c["category"] for c in r["categories"][:8]])
                st.bar_chart(cats, horizontal=True, color=NAVY)
            for mc in r["month_changes"][:4]:
                arrow = "🔺" if mc["change"] > 0 else "🔻"
                st.markdown(f"{arrow} **{esc(mc['category'])}**: {inr(mc['previous'])} → {inr(mc['current'])}", unsafe_allow_html=True)
                _evidence_lines(mc["drivers"][:2])
        with tabs[4]:
            for n, a in enumerate(r["actions"], 1):
                st.markdown(f"**{n}. {esc(a['title'])}**" + (f" · {L('save', 'बचत')} {inr(a['saving'])}" if a.get("saving") else ""),
                            unsafe_allow_html=True)
                st.caption(a["detail"])
            if r["card_summary"]:
                cs = r["card_summary"]
                st.caption(L(f"Card: total due {inr(cs.get('card_total_due') or 0, 2)}, minimum {inr(cs.get('card_minimum_due') or 0, 2)}, "
                             f"due {cs.get('card_due_date')}", f"कार्ड: कुल बकाया {inr(cs.get('card_total_due') or 0, 2)}, न्यूनतम "
                             f"{inr(cs.get('card_minimum_due') or 0, 2)}, अंतिम तारीख {cs.get('card_due_date')}"))
        with tabs[5]:
            if txns:
                df = pd.DataFrame([{"ID": t["txn_id"], L("Date", "तारीख"): t["date"], L("Merchant", "व्यापारी"): t["merchant"],
                                    L("Amount", "राशि"): (-1 if t["type"] == "debit" else 1) * t["amount"],
                                    L("Category", "श्रेणी"): t["category"],
                                    L("Flags", "संकेत"): " ".join(f for f, on in (("🔁", t["recurring"]), ("⚠️dup", t["duplicate"]),
                                                                            ("💸fee", t["fee"])) if on),
                                    L("Source", "स्रोत"): t["source"]} for t in txns])
                event = st.dataframe(df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                                     key=f"txn_tbl_{key}")
                rows = event.selection.rows if event and getattr(event, "selection", None) else []
                st.caption(L("Click a row, then ask Saathi to explain it.", "किसी पंक्ति पर क्लिक करें, फिर साथी से समझाने को कहें।"))
                if rows and st.button(L(f"💬 Explain transaction {df.iloc[rows[0]]['ID']}", f"💬 लेन-देन {df.iloc[rows[0]]['ID']} समझाइए"),
                                      key=f"explain_{key}"):
                    ask(f"Explain transaction {df.iloc[rows[0]]['ID']}")
        st.caption("⚠️ " + r["disclaimer"])


def _loan_card(r: dict, key: str, ask) -> None:
    with st.container(border=True):
        i = r["inputs"]
        st.markdown(f"**{L('🏦 Loan Pre-Flight', '🏦 लोन प्री-फ्लाइट')}** · {inr(i['loan_amount'])} · {i['tenure_months']} "
                    f"{L('months', 'महीने')} · {i['annual_rate_pct']}%")
        status = r["status"]
        m = st.columns(4)
        m[0].metric(L("EMI", "ईएमआई"), inr(r["emi"]))
        m[1].metric(L("Total interest", "कुल ब्याज"), inr(r["total_interest"]))
        m[2].metric(L("Left each month", "हर महीने बचेगा"), inr(r["after"]["buffer"]), delta=inr(r["after"]["buffer"] - r["before"]["buffer"]))
        m[3].markdown(f"{L('EMI stress test', 'ईएमआई स्ट्रेस टेस्ट')}<br>{pill(lending.STATUS_LABELS[status], STATUS_KIND[status])}",
                      unsafe_allow_html=True)
        (note if status == "safe" else alert)(r["status_reason"])
        chart = pd.DataFrame({
            L("Existing EMIs", "मौजूदा ईएमआई"): [r["before"]["existing_emis"], r["after"]["existing_emis"]],
            L("Card dues", "कार्ड बकाया"): [r["before"]["card_dues"], r["after"]["card_dues"]],
            L("Expenses", "खर्च"): [r["before"]["expenses"], r["after"]["expenses"]],
            L("New EMI", "नई ईएमआई"): [0, r["after"]["new_emi"]],
            L("Left over", "बचत"): [max(r["before"]["buffer"], 0), max(r["after"]["buffer"], 0)],
        }, index=[L("Before loan", "लोन से पहले"), L("After loan", "लोन के बाद")])
        st.markdown(f"**{L('Monthly cash flow: before vs after', 'मासिक नकदी: पहले बनाम बाद')}**")
        st.bar_chart(chart, horizontal=True, color=[NAVY, "#5B6B82", "#A7B4C6", CYAN, "#12A26B"])
        st.dataframe(pd.DataFrame([{L("Scenario", "परिस्थिति"): s["name"], L("Detail", "विवरण"): s["detail"],
                                    L("EMI", "ईएमआई"): inr(s["emi"]), L("Left over", "बचत"): inr(s["buffer"]),
                                    L("Result", "नतीजा"): lending.STATUS_LABELS[s["status"]]} for s in r["scenarios"]]),
                     hide_index=True, width="stretch")
        if r["risk_factors"]:
            st.markdown(f"**{L('Possible rejection risks (from your numbers)', 'संभावित अस्वीकृति कारण (आपकी संख्याओं से)')}**")
            for f in r["risk_factors"]:
                st.markdown(f"- **{esc(f['title'])}** {pill(f['severity'], {'high': 'bad', 'medium': 'warn', 'low': 'info'}[f['severity']])} - {esc(f['detail'])}",
                            unsafe_allow_html=True)
        st.markdown(f"**{L('What would help', 'क्या मदद करेगा')}**")
        for a in r["actions"][:5]:
            st.markdown(f"- {esc(a)}", unsafe_allow_html=True)
        with st.expander(L("Documents checklist and loan terms explained", "दस्तावेज़ सूची और लोन शर्तें")):
            for d in r["documents"]:
                st.markdown(f"- {L(d['en'], d.get('hi', d['en']))}")
            for k, v in r["explanations"].items():
                st.markdown(f"**{k.replace('_', ' ').title()}** - {v}")
        if r["assumptions"]:
            st.caption(" ".join(r["assumptions"]))
        c = st.columns(3)
        for n, (en, hi) in enumerate([("What happens if I choose a 5-year tenure instead?", "अगर मैं 5 साल की अवधि चुनूँ तो क्या होगा?"),
                                      ("Can I afford this loan if a family member is hospitalised next month?", "अगर अगले महीने परिवार में किसी को अस्पताल जाना पड़े तो क्या मैं यह लोन चुका पाऊँगा?"),
                                      ("Why might my loan application be rejected?", "मेरा लोन आवेदन क्यों अस्वीकार हो सकता है?")]):
            if c[n].button(L(en, hi), key=f"loanq_{key}_{n}", width="stretch"):
                ask(L(en, hi))
        st.caption("⚠️ " + r["disclaimer"] + " " + L("General educational guidance based on the numbers you entered.",
                                                    "आपकी दी गई संख्याओं पर आधारित सामान्य शैक्षिक जानकारी।"))


def _rejection_card(d: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{L('📄 What your document says', '📄 आपका दस्तावेज़ क्या कहता है')}** · {esc(d['source'])}", unsafe_allow_html=True)
        if not d["reasons"]:
            note(d["message"])
        for n, reason in enumerate(d["reasons"], 1):
            st.markdown(f"**{n}. {esc(reason['title'])}**", unsafe_allow_html=True)
            st.caption(f"[D{n}] “{reason['quote']}” - {reason['location']}")
            st.markdown(esc(reason["explanation"]), unsafe_allow_html=True)
            for a in reason["actions"]:
                st.markdown(f"- {esc(a)}", unsafe_allow_html=True)
        st.caption("⚠️ " + d["disclaimer"])
