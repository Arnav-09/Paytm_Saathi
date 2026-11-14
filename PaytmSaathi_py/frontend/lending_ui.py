"""Loans journey: EMI + prepayment planner, eligibility + levers, true-cost comparison, documents & terms."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ai_engine import get_llm_provider, lending
from frontend.common import L, lang, pick, run
from frontend.theme import CYAN, NAVY, esc, inr, note, page_header, pill

LOAN_LABELS = {"personal": ("Personal", "पर्सनल"), "home": ("Home", "होम"), "vehicle": ("Vehicle", "वाहन"),
               "business": ("Business", "बिज़नेस"), "education": ("Education", "शिक्षा"), "gold": ("Gold", "गोल्ड")}
EMP_LABELS = {"salaried": ("Salaried", "नौकरीपेशा"), "self_employed": ("Self-employed", "स्वरोज़गार")}


def _loan(k):
    return LOAN_LABELS[k][lang() == "hi"]


def _emp(k):
    return EMP_LABELS[k][lang() == "hi"]


def _years(months: int) -> str:
    y, m = divmod(months, 12)
    if lang() == "hi":
        return f"{y} साल {m} महीने" if m else f"{y} साल"
    return f"{y} yr {m} mo" if m else f"{y} yr"


def tab_emi():
    a, b, c = st.columns(3)
    principal = a.number_input(L("Loan amount (₹)", "लोन राशि (₹)"), 10000.0, 100000000.0, 2500000.0, 50000.0, key="emi_p")
    rate = b.number_input(L("Interest rate (% per year)", "ब्याज दर (% सालाना)"), 0.0, 60.0, 8.75, 0.25, key="emi_r")
    years = c.number_input(L("Tenure (years)", "अवधि (साल)"), 0.5, 30.0, 20.0, 0.5, key="emi_t")
    months = int(round(years * 12))
    s = run(lending.loan_summary, principal, rate, months)
    if not s:
        return
    m = st.columns(4)
    m[0].metric(L("Monthly EMI", "मासिक ईएमआई"), inr(s.emi))
    m[1].metric(L("Total interest", "कुल ब्याज"), inr(s.total_interest))
    m[2].metric(L("Total payment", "कुल भुगतान"), inr(s.total_payment))
    m[3].metric(L("Interest share", "ब्याज का हिस्सा"), f"{s.interest_share_pct}%")
    if s.interest_share_pct >= 40:
        note(L(f"You'd pay {inr(s.total_interest)} in interest - {s.interest_share_pct}% of everything you repay. A prepayment plan below can cut this a lot.",
               f"आप {inr(s.total_interest)} ब्याज देंगे - कुल भुगतान का {s.interest_share_pct}%। नीचे का प्रीपेमेंट प्लान इसे काफ़ी घटा सकता है।"))

    st.markdown(f"#### {L('💡 Prepayment planner', '💡 प्रीपेमेंट प्लानर')}")
    st.caption(L("Pay a little extra and watch your loan finish years earlier. Your EMI stays the same.",
                 "थोड़ा ज़्यादा चुकाएँ और देखें लोन कितने साल पहले खत्म होता है। आपकी ईएमआई वही रहती है।"))
    p1, p2, p3 = st.columns(3)
    extra_m = p1.number_input(L("Extra every month (₹)", "हर महीने अतिरिक्त (₹)"), 0.0, 10000000.0, 0.0, 500.0, key="pp_m")
    extra_y = p2.number_input(L("Extra once a year, e.g. bonus (₹)", "साल में एक बार अतिरिक्त, जैसे बोनस (₹)"), 0.0, 100000000.0, 50000.0, 5000.0, key="pp_y")
    lump = p3.number_input(L("One-time lump sum (₹)", "एकमुश्त राशि (₹)"), 0.0, 100000000.0, 0.0, 10000.0, key="pp_l")
    plan = run(lending.prepayment_plan, principal, rate, months, extra_m, extra_y, lump, 12)
    if plan:
        k = st.columns(3)
        k[0].metric(L("Interest saved", "बचा ब्याज"), inr(plan.interest_saved))
        k[1].metric(L("Loan ends earlier by", "लोन इतना पहले खत्म"), _years(plan.months_saved) if plan.months_saved else "—")
        k[2].metric(L("New tenure", "नई अवधि"), _years(plan.new_months))
        n = max(len(plan.balance_original), len(plan.balance_new))
        chart = pd.DataFrame({
            L("Without prepayment", "प्रीपेमेंट के बिना"): plan.balance_original + [0.0] * (n - len(plan.balance_original)),
            L("With prepayment", "प्रीपेमेंट के साथ"): plan.balance_new + [0.0] * (n - len(plan.balance_new)),
        }, index=pd.Index(range(1, n + 1), name=L("Year", "साल")))
        st.line_chart(chart, color=[NAVY, CYAN], y_label=L("Loan balance (₹)", "बकाया लोन (₹)"))
        st.caption(L(plan.note, "मान्यता: प्रीपेमेंट से अवधि घटती है और ईएमआई वही रहती है। फिक्स्ड रेट लोन पर कुछ लेंडर प्रीपेमेंट शुल्क लेते हैं; व्यक्तियों के फ्लोटिंग रेट होम लोन पर आमतौर पर नहीं।"))

    with st.expander(L("Month-by-month schedule", "महीने-दर-महीने शेड्यूल")):
        df = pd.DataFrame([r.model_dump() for r in lending.amortization_schedule(principal, rate, months)])
        st.dataframe(df, hide_index=True, width="stretch")


def tab_eligibility():
    with st.form("elig"):
        a, b, c = st.columns(3)
        ltype = a.selectbox(L("Loan type", "लोन का प्रकार"), list(LOAN_LABELS), format_func=_loan)
        emp = b.selectbox(L("Employment", "रोज़गार"), list(EMP_LABELS), format_func=_emp)
        age = c.number_input(L("Age", "उम्र"), 18, 80, 30)
        d, e, f = st.columns(3)
        income = d.number_input(L("Monthly income (₹)", "मासिक आय (₹)"), 0.0, 10000000.0, 60000.0, 1000.0)
        emi_now = e.number_input(L("Existing EMIs per month (₹)", "मौजूदा मासिक ईएमआई (₹)"), 0.0, 10000000.0, 8000.0, 500.0)
        score = f.number_input(L("Credit score (0 = don't know)", "क्रेडिट स्कोर (0 = पता नहीं)"), 0, 900, 0)
        g, h = st.columns(2)
        amount = g.number_input(L("Loan amount you need (₹)", "आपको चाहिए लोन राशि (₹)"), 1000.0, 100000000.0, 800000.0, 10000.0)
        tenure = h.number_input(L("Tenure (months)", "अवधि (महीने)"), 3, 360, 36)
        go = st.form_submit_button(L("Check my eligibility", "मेरी पात्रता जाँचें"), type="primary")
    if go:
        st.session_state["elig_applicant"] = lending.LoanApplicant(
            loan_type=ltype, employment_type=emp, age=int(age), monthly_income=income, existing_emi=emi_now,
            credit_score=int(score) or None, requested_amount=amount, tenure_months=int(tenure))
    applicant = st.session_state.get("elig_applicant")
    if not applicant:
        st.caption(L("This is a soft check: nothing is shared with lenders and your credit score is not affected.",
                     "यह एक सॉफ्ट जाँच है: लेंडर को कुछ नहीं भेजा जाता और आपके क्रेडिट स्कोर पर असर नहीं पड़ता।"))
        return
    res = run(lending.check_loan_eligibility, applicant)
    if not res:
        return
    kind, label = {"likely_eligible": ("ok", L("Likely eligible", "पात्र होने की संभावना")),
                   "partially_eligible": ("warn", L("Partly eligible", "आंशिक रूप से पात्र")),
                   "unlikely": ("bad", L("Unlikely right now", "अभी संभावना कम"))}[res.decision]
    st.markdown(pill(label, kind), unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric(L("You could borrow about", "आप लगभग ले सकते हैं"), inr(res.max_eligible_amount))
    m[1].metric(L("Indicative rate", "संभावित दर"), f"{res.indicative_rate_pct}%" if res.indicative_rate_pct else "—")
    m[2].metric(L("EMI for your request", "आपकी माँग की ईएमआई"), inr(res.emi_for_requested))
    m[3].metric(L("Income going to EMIs", "आय का किस्तों में हिस्सा"),
                f"{res.foir_after_loan_pct}%" if res.foir_after_loan_pct is not None else "—")
    for text in pick(res.reasons, res.reasons_hi):
        st.markdown(f"- {text}")

    st.markdown(f"#### {L('What would change the answer?', 'क्या बदलने से जवाब बदलेगा?')}")
    levers = run(lending.eligibility_levers, applicant)
    if levers:
        st.dataframe(pd.DataFrame([{
            L("If you…", "अगर आप…"): L(lv.lever, lv.lever_hi),
            L("You could borrow", "आप ले सकते हैं"): inr(lv.max_eligible_amount),
            L("Change", "बदलाव"): ("+" + inr(lv.change)) if lv.change > 0 else ("—" if lv.change == 0 else inr(lv.change)),
        } for lv in levers]), hide_index=True, width="stretch")
    tips = pick(res.tips, res.tips_hi)
    for text in tips:
        note(text)
    with st.expander(L("Documents you will need", "ज़रूरी दस्तावेज़")):
        for doc in res.documents:
            st.markdown("- " + doc["hi" if lang() == "hi" else "en"])
    st.caption(res.disclaimer)


def tab_compare():
    note(L("The lowest EMI is not always the cheapest loan. We add fees to show the real cost.",
           "सबसे कम ईएमआई हमेशा सबसे सस्ता लोन नहीं होता। असली लागत दिखाने के लिए हम शुल्क भी जोड़ते हैं।"))
    n = st.radio(L("Number of offers", "ऑफर की संख्या"), [2, 3], horizontal=True, key="cmp_n")
    defaults = [("Lender A", 11.0, 1.0, 36), ("Lender B", 10.5, 3.0, 36), ("Lender C", 12.0, 0.0, 48)]
    offers = []
    for i, col in enumerate(st.columns(n)):
        with col.container(border=True):
            name = st.text_input(L("Name", "नाम"), defaults[i][0], key=f"cmp_name{i}")
            p = st.number_input(L("Amount (₹)", "राशि (₹)"), 10000.0, 100000000.0, 500000.0, 10000.0, key=f"cmp_p{i}")
            r = st.number_input(L("Rate (% p.a.)", "दर (% सालाना)"), 0.0, 60.0, defaults[i][1], 0.25, key=f"cmp_r{i}")
            t = st.number_input(L("Tenure (months)", "अवधि (महीने)"), 3, 360, defaults[i][3], key=f"cmp_t{i}")
            fee = st.number_input(L("Processing fee (%)", "प्रोसेसिंग शुल्क (%)"), 0.0, 10.0, defaults[i][2], 0.25, key=f"cmp_f{i}")
            offers.append(lending.LoanOffer(name=name or f"Offer {i + 1}", principal=p, annual_rate_pct=r,
                                            tenure_months=int(t), processing_fee_pct=fee))
    results = run(lending.compare_loan_offers, offers)
    if not results:
        return
    left, right = st.columns([1.2, 1])
    left.dataframe(pd.DataFrame([{
        L("Offer", "ऑफर"): r.name + (" ⭐" if r.best else ""), L("EMI", "ईएमआई"): inr(r.emi),
        L("Interest", "ब्याज"): inr(r.total_interest), L("Fees", "शुल्क"): inr(r.fees),
        L("Total cost", "कुल लागत"): inr(r.total_cost), L("Effective rate", "प्रभावी दर"): f"{r.effective_apr_pct}%"}
        for r in results]), hide_index=True, width="stretch")
    right.bar_chart(pd.DataFrame({L("Interest", "ब्याज"): [r.total_interest for r in results],
                                  L("Fees", "शुल्क"): [r.fees for r in results]}, index=[r.name for r in results]),
                    color=[NAVY, CYAN], horizontal=True, x_label=L("Total cost (₹)", "कुल लागत (₹)"))
    best = next(r for r in results if r.best)
    lowest_emi = min(results, key=lambda r: r.emi)
    st.success(L(f"{best.name} costs the least overall: {inr(best.total_cost)} in interest and fees.",
                 f"{best.name} की कुल लागत सबसे कम है: {inr(best.total_cost)} ब्याज और शुल्क में।"))
    if lowest_emi.name != best.name:
        st.warning(L(f"{lowest_emi.name} has the lowest EMI, but costs {inr(lowest_emi.total_cost - best.total_cost)} more overall.",
                     f"{lowest_emi.name} की ईएमआई सबसे कम है, पर कुल मिलाकर {inr(lowest_emi.total_cost - best.total_cost)} ज़्यादा महँगा है।"))


def tab_help():
    a, b = st.columns(2)
    ltype = a.selectbox(L("Loan type", "लोन का प्रकार"), list(LOAN_LABELS), format_func=_loan, key="doc_lt")
    emp = b.selectbox(L("Employment", "रोज़गार"), list(EMP_LABELS), format_func=_emp, key="doc_emp")
    docs = lending.required_documents(ltype, emp)
    done = 0
    with st.container(border=True):
        st.markdown(f"**{L('Document checklist', 'दस्तावेज़ चेकलिस्ट')}**")
        for i, doc in enumerate(docs):
            done += st.checkbox(doc["hi" if lang() == "hi" else "en"], key=f"doc_{ltype}_{emp}_{i}")
        st.progress(done / len(docs), text=L(f"{done} of {len(docs)} ready", f"{len(docs)} में से {done} तैयार"))
    st.markdown(f"#### {L('Understand a loan offer or agreement', 'लोन ऑफर या एग्रीमेंट समझें')}")
    text = st.text_area(L("Paste the text from the offer or agreement", "ऑफर या एग्रीमेंट का टेक्स्ट पेस्ट करें"), height=130, key="loan_text")
    if st.button(L("Explain it", "समझाएँ"), disabled=not text.strip(), key="loan_explain", type="primary"):
        res = run(lending.explain_loan_text, text, lang(), get_llm_provider())
        if res:
            if res["explanation"]:
                st.markdown(f'<div class="ps-note">{esc(res["explanation"]).replace(chr(10), "<br>")}</div>',
                            unsafe_allow_html=True)
            elif res["note"]:
                note(L(res["note"], "पूरी व्याख्या के लिए एआई कुंजी जोड़ें (.env देखें)।"))
            for term in res["terms"]:
                st.markdown(f"**{term['term']}**: {term['hi' if lang() == 'hi' else 'en']}")
    with st.expander(L("📖 Loan words in plain language", "📖 लोन के शब्द आसान भाषा में")):
        for term, info in lending.GLOSSARY.items():
            st.markdown(f"**{term}** - {info['hi' if lang() == 'hi' else 'en']}")


def render():
    page_header(L("Loans", "लोन"), L("Plan EMIs, see what you qualify for, and compare offers by their real cost.",
                                     "ईएमआई प्लान करें, पात्रता देखें और ऑफर की असली लागत से तुलना करें।"))
    tabs = st.tabs([L("🧮 EMI & prepayment", "🧮 ईएमआई और प्रीपेमेंट"), L("✅ Eligibility", "✅ पात्रता"),
                    L("⚖️ Compare offers", "⚖️ ऑफर तुलना"), L("📑 Documents & terms", "📑 दस्तावेज़ और शर्तें")])
    for tab, fn in zip(tabs, (tab_emi, tab_eligibility, tab_compare, tab_help)):
        with tab:
            fn()
