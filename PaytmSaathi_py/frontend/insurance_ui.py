"""Insurance journey: policy overview, Claim Pre-Flight, and clause explanations."""

from __future__ import annotations

import re
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from ai_engine import (
    ClaimContext,
    CashlessEligibility,
    ClaimReadinessReport,
    FinancialEstimate,
    HospitalBill,
    PolicyCard,
    PolicyCitation,
    PreAuthRequirement,
    ReadinessScore,
    RoomRentImpact,
    accounts,
    estimate_claim_payout,
    explain_clause_text,
    explain_policy_clause,
    extract_hospital_bill,
    generate_claim_readiness_report,
    get_claim_relevant_clauses,
    get_policy_details,
    process_policy,
)
from ai_engine.bill_extractor import CATEGORIES
from ai_engine.policy_insights import (
    claim_documents,
    coverage_at_a_glance,
    room_rent_what_if,
    waiting_period_timeline,
)
from ai_engine.schemas import BillLineItem
from frontend.common import (
    L, bi, current_policy, demo_mode, lang, my_policies, need_policy, pair, pick, policy_key,
    policy_label, run, safe_name, uid,
)
from frontend.theme import alert, empty_state, esc, inr, note, page_header, pill, steps

CATEGORY_LABELS = {
    "room_rent": ("Room rent", "कमरे का किराया"), "icu_rent": ("ICU", "आईसीयू"),
    "doctor_fees": ("Doctor fees", "डॉक्टर फीस"), "surgery": ("Surgery / OT", "सर्जरी / ओटी"),
    "nursing": ("Nursing", "नर्सिंग"), "medicines": ("Medicines", "दवाइयाँ"),
    "investigations": ("Tests", "जाँचें"), "consumables": ("Consumables", "उपभोग्य सामग्री"),
    "non_payable": ("Non-payable", "भुगतान योग्य नहीं"), "other": ("Other", "अन्य"),
}


def _policy_start(pid: str) -> date | None:
    return st.session_state.get(f"pstart_{pid}")


# ---------------------------------------------------------------------------
# My policy
# ---------------------------------------------------------------------------
def tab_policy():
    mine = my_policies()
    left, right = st.columns([2, 1], vertical_alignment="bottom")
    if mine:
        idx = mine.index(current_policy()) if current_policy() in mine else 0
        st.session_state["policy_id"] = left.selectbox(
            L("Your policies", "आपकी पॉलिसी"), mine, index=idx, format_func=policy_label, key="pol_pick")
    with right.popover(L("➕ Upload a policy", "➕ पॉलिसी अपलोड करें"), width="stretch"):
        _upload_policy()
    pid = current_policy()
    if not pid:
        empty_state("📄", L("Upload your health policy to begin", "शुरू करने के लिए अपनी हेल्थ पॉलिसी अपलोड करें"),
                    L("PDF from your insurer's email or app. Scanned copies work too. We'll show what's covered, what isn't, and the fine print that affects claims.",
                      "बीमा कंपनी के ईमेल या ऐप से मिली PDF। स्कैन कॉपी भी चलेगी। हम बताएँगे क्या कवर है, क्या नहीं, और क्लेम पर असर डालने वाली बारीक शर्तें।"))
        return
    details = run(get_policy_details, pid)
    if not details:
        return
    st.markdown(f"### {esc(details.policy_name or policy_label(pid))}")
    st.caption(details.insurer_name or L("Insurer name not found in the document", "दस्तावेज़ में बीमा कंपनी का नाम नहीं मिला"))
    m = st.columns(4)
    m[0].metric(L("Sum insured", "बीमित राशि"), inr(details.sum_insured.value) if details.sum_insured else "—")
    m[1].metric(L("Room rent limit", "कमरे के किराए की सीमा"), inr(details.room_rent_limit.value) if details.room_rent_limit else L("No cap found", "सीमा नहीं मिली"))
    m[2].metric(L("Co-pay", "सह-भुगतान"), f"{details.copay_percentage:g}%" if details.copay_percentage is not None else L("None found", "नहीं मिला"))
    m[3].metric(L("Exclusions", "अपवर्जन"), len(details.exclusions))

    st.markdown(f"#### {L('Coverage at a glance', 'एक नज़र में कवरेज')}")
    g = coverage_at_a_glance(details)
    cols = st.columns(3)
    for col, title, items, cls, empty in [
        (cols[0], L("✅ Covered", "✅ कवर है"), g.covered, "ok", L("Nothing extracted yet", "अभी कुछ नहीं मिला")),
        (cols[1], L("❌ Not covered", "❌ कवर नहीं"), g.not_covered, "bad", L("No exclusions extracted", "कोई अपवर्जन नहीं मिला")),
        (cols[2], L("⚠️ Watch-outs", "⚠️ ध्यान दें"), g.watch_outs, "warn", L("No limits extracted", "कोई सीमा नहीं मिली")),
    ]:
        rows = "".join(
            f'<div class="item">{esc(L(i.text, i.text_hi))}'
            + (f'<div class="ref">{esc(L("Clause", "क्लॉज़"))} {esc(i.clause or "-")} · {esc(L("page", "पेज"))} {i.page}</div>' if i.page else "")
            + "</div>" for i in items) or f'<div class="item" style="color:#5B6B82">{esc(empty)}</div>'
        col.markdown(f'<div class="ps-glance ps-g-{cls}"><h5>{title}</h5>{rows}</div>', unsafe_allow_html=True)
    if g.unknown:
        st.caption(L("Not found in this document: ", "इस दस्तावेज़ में नहीं मिला: ") + ", ".join(g.unknown))

    st.markdown(f"#### {L('Waiting-period countdown', 'प्रतीक्षा अवधि काउंटडाउन')}")
    start = st.date_input(L("When did this policy start?", "यह पॉलिसी कब शुरू हुई?"), value=_policy_start(pid),
                          max_value=date.today(), key=f"pstart_in_{pid}", format="DD/MM/YYYY")
    if start:
        st.session_state[f"pstart_{pid}"] = start
        rows = run(waiting_period_timeline, details, start)
        if rows:
            for r in rows:
                c1, c2 = st.columns([3, 2], vertical_alignment="center")
                c1.progress(r.progress, text=f"**{r.condition}** · {r.duration_days} {L('days', 'दिन')}")
                c2.markdown(pill(L("Completed", "पूरी हुई"), "ok") if r.completed else
                            pill(L(f"Ends {r.ends_on} · {r.days_left} days left", f"{r.ends_on} को खत्म · {r.days_left} दिन बाकी"), "warn"),
                            unsafe_allow_html=True)
        elif rows is not None:
            st.caption(L("No waiting periods with a day count were found in this policy.", "इस पॉलिसी में दिनों वाली कोई प्रतीक्षा अवधि नहीं मिली।"))
    else:
        st.caption(L("Add the start date to see exactly when each waiting period ends.", "हर प्रतीक्षा अवधि कब खत्म होगी, यह देखने के लिए शुरू होने की तारीख डालें।"))
    with st.expander(L("All extracted details (JSON)", "निकाली गई पूरी जानकारी (JSON)")):
        st.json(details.model_dump())


def _upload_policy():
    up = st.file_uploader(L("Policy PDF (text or scanned)", "पॉलिसी PDF (टेक्स्ट या स्कैन)"), type=["pdf"], key="pol_up")
    default = re.sub(r"[^a-z0-9_]+", "_", Path(up.name).stem.lower()).strip("_")[:30] if up else ""
    name = st.text_input(L("Give it a short name", "इसे एक छोटा नाम दें"), value=default, key=f"pol_name_{default}")
    if st.button(L("Read my policy", "मेरी पॉलिसी पढ़ें"), type="primary", disabled=not (up and name.strip()), key="pol_go"):
        pid = policy_key(name.strip())
        with st.spinner(L("Reading your policy - the first time can take a minute…", "आपकी पॉलिसी पढ़ी जा रही है - पहली बार थोड़ा समय लग सकता है…")):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / safe_name(up.name)
                path.write_bytes(up.getvalue())
                details = run(process_policy, str(path), pid)
        if details:
            accounts.link_policy(uid(), pid)
            st.session_state["policy_id"] = pid
            st.rerun()


# ---------------------------------------------------------------------------
# Claim Pre-Flight
# ---------------------------------------------------------------------------
def _bill_to_df(bill: HospitalBill) -> pd.DataFrame:
    rows = [{"description": i.description, "category": i.category, "quantity": i.quantity,
             "amount": i.amount, "is_payable": i.is_payable} for i in bill.line_items]
    return pd.DataFrame(rows or [{"description": "", "category": "other", "quantity": None,
                                  "amount": 0.0, "is_payable": True}])


def _set_bill(bill: HospitalBill) -> None:
    st.session_state["bill"] = bill
    st.session_state.pop("estimate", None)
    st.session_state["bill_rev"] = st.session_state.get("bill_rev", 0) + 1


def _items_from(edited: pd.DataFrame) -> list[BillLineItem]:
    items = []
    for row in edited.to_dict("records"):
        desc = str(row.get("description") or "").strip()
        if not desc:
            continue
        qty, amount = row.get("quantity"), row.get("amount")
        items.append(BillLineItem(
            description=desc, category=row.get("category") or "other",
            quantity=None if qty is None or pd.isna(qty) else float(qty),
            amount=0.0 if amount is None or pd.isna(amount) else float(amount),
            is_payable=bool(row.get("is_payable", True))))
    return items


def tab_claim():
    if not need_policy():
        return
    pid = current_policy()
    bill: HospitalBill | None = st.session_state.get("bill")
    stage = 0 if bill is None else (2 if st.session_state.get("estimate") else 1)
    steps([L("Add hospital bill", "अस्पताल का बिल जोड़ें"), L("Check the details", "विवरण जाँचें"),
           L("See your payout", "अपना भुगतान देखें")], stage)

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    up = c1.file_uploader(L("Hospital bill or estimate (PDF, scan or photo)", "अस्पताल का बिल या एस्टिमेट (PDF, स्कैन या फोटो)"),
                          type=["pdf", "png", "jpg", "jpeg"], key="bill_up")
    if c2.button(L("Read bill", "बिल पढ़ें"), type="primary", disabled=up is None, key="bill_go", width="stretch"):
        with st.spinner(L("Reading bill…", "बिल पढ़ा जा रहा है…")):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / safe_name(up.name)
                path.write_bytes(up.getvalue())
                new_bill = run(extract_hospital_bill, str(path))
        if new_bill:
            _set_bill(new_bill)
            st.rerun()
    if c2.button(L("Enter manually", "हाथ से भरें"), key="bill_manual", width="stretch"):
        _set_bill(HospitalBill(extraction_method="manual"))
        st.rerun()
    st.caption(L("Tip: ask the hospital for a cost estimate before admission and check it here - you'll know your out-of-pocket amount in advance.",
                 "सुझाव: भर्ती से पहले अस्पताल से खर्च का एस्टिमेट लें और यहाँ जाँचें - आपको पहले से पता चलेगा कि अपनी जेब से कितना देना होगा।"))
    if bill is None:
        return
    for w in bill.warnings:
        st.warning(w)

    rev = st.session_state.get("bill_rev", 0)
    st.markdown(f"#### {L('Check the bill', 'बिल जाँचें')}")
    edited = st.data_editor(
        _bill_to_df(bill), num_rows="dynamic", width="stretch", key=f"bill_ed_{rev}",
        column_config={
            "description": st.column_config.TextColumn(L("Item", "आइटम"), width="large"),
            "category": st.column_config.SelectboxColumn(L("Category", "श्रेणी"), options=CATEGORIES,
                                                         format_func=lambda c: pair(CATEGORY_LABELS.get(c, (c, c)))),
            "quantity": st.column_config.NumberColumn(L("Qty / days", "मात्रा/दिन"), min_value=0),
            "amount": st.column_config.NumberColumn(L("Amount (₹)", "राशि (₹)"), min_value=0, format="%.2f"),
            "is_payable": st.column_config.CheckboxColumn(L("Payable?", "भुगतान योग्य?")),
        })
    a, b, c = st.columns(3)
    days = a.number_input(L("Days in hospital", "अस्पताल में दिन"), min_value=0, value=int(bill.days_of_stay or 0), key=f"days_{rev}")
    diagnosis = b.text_input(L("Diagnosis", "बीमारी"), value=bill.diagnosis or "", key=f"dx_{rev}")
    procedure = c.text_input(L("Procedure / surgery", "प्रक्रिया / सर्जरी"), value=bill.procedure or "", key=f"proc_{rev}")
    d, e, f = st.columns(3)
    start = _policy_start(pid)
    if start:
        d.markdown(f"**{L('Policy started', 'पॉलिसी शुरू')}**  \n{start:%d %b %Y} · {(date.today() - start).days} {L('days ago', 'दिन पहले')}")
        age_days = (date.today() - start).days
    else:
        age_days = d.number_input(L("Policy age (days)", "पॉलिसी की उम्र (दिन)"), min_value=0, value=0, key=f"age_{rev}",
                                  help=L("0 = unknown. Tip: add the start date in 'My policy'.", "0 = पता नहीं। सुझाव: 'मेरी पॉलिसी' में शुरू होने की तारीख डालें।"))
    prior = e.number_input(L("Already claimed this year (₹)", "इस साल पहले से क्लेम (₹)"), min_value=0.0, value=0.0, key=f"prior_{rev}")
    ped = f.checkbox(L("Condition existed before the policy", "बीमारी पॉलिसी से पहले की है"), key=f"ped_{rev}")

    if st.button(L("Calculate my payout", "मेरा भुगतान निकालें"), type="primary", key=f"calc_{rev}"):
        items = _items_from(edited)
        final = bill.model_copy(update={
            "line_items": items, "days_of_stay": int(days) or None, "diagnosis": diagnosis or None,
            "procedure": procedure or None, "computed_total": round(sum(i.amount for i in items), 2)})
        ctx = ClaimContext(policy_age_days=int(age_days) or None, previously_claimed=float(prior), is_pre_existing=ped)
        with st.spinner(L("Calculating…", "गणना हो रही है…")):
            est = run(estimate_claim_payout, pid, final, ctx)
        if est:
            st.session_state["estimate"] = (est, final, ctx)
            st.rerun()
    result = st.session_state.get("estimate")
    if result:
        render_estimate(pid, *result)


def render_estimate(pid, est, bill: HospitalBill, ctx: ClaimContext):
    st.divider()
    kind, label = {"likely_payable": ("ok", L("Likely payable", "भुगतान की संभावना")),
                   "needs_review": ("warn", L("Needs review", "समीक्षा ज़रूरी")),
                   "likely_excluded": ("bad", L("Likely not payable", "भुगतान की संभावना कम"))}[est.eligibility_status]
    st.markdown(f"#### {L('Your estimate', 'आपका अनुमान')} &nbsp; {pill(label, kind)}", unsafe_allow_html=True)
    m = st.columns(3)
    m[0].metric(L("Hospital bill", "अस्पताल का बिल"), inr(est.bill_total))
    m[1].metric(L("Insurer may pay", "बीमा कंपनी दे सकती है"), inr(est.estimated_payout))
    m[2].metric(L("You may pay", "आपको देना पड़ सकता है"), inr(est.patient_pays))
    if est.bill_total:
        st.progress(min(est.estimated_payout / est.bill_total, 1.0),
                    text=L(f"Insurance covers about {est.estimated_payout / est.bill_total:.0%} of this bill",
                           f"बीमा इस बिल का लगभग {est.estimated_payout / est.bill_total:.0%} कवर करता है"))
    if est.eligibility_status != "likely_payable" and est.payout_if_covered:
        st.caption(L(f"If this claim is covered, the estimate is {inr(est.payout_if_covered)}.",
                     f"अगर यह क्लेम कवर होता है तो अनुमान {inr(est.payout_if_covered)} है।"))
    for text in pick(est.flags, est.flags_hi):
        alert(text)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(f"**{L('How we worked it out', 'हिसाब कैसे बना')}**")
        st.dataframe(pd.DataFrame([{
            L("Step", "चरण"): L(s.description, s.description_hi or s.description),
            L("Deducted", "कटौती"): inr(s.deduction) if s.deduction else "—",
            L("Running total", "बची राशि"): inr(s.amount_after),
            L("Clause", "क्लॉज़"): ((s.clause_reference or "") + (f" (p.{s.source_page})" if s.source_page else "")).strip() or "—",
        } for s in est.steps]), hide_index=True, width="stretch")
    with right:
        st.markdown(f"**{L('What if you chose a different room?', 'अगर आप दूसरा कमरा चुनते तो?')}**")
        details = get_policy_details(pid)
        options = None
        if details.room_rent_limit and any(i.category == "room_rent" for i in bill.line_items):
            try:
                options = room_rent_what_if(details, bill, ctx)
            except Exception:
                options = None
        if options:
            labels = {"Your room": L("Your room", "आपका कमरा"), "Within limit": L("Within limit", "सीमा के भीतर"),
                      "Costlier room": L("Costlier room", "महँगा कमरा")}
            df = pd.DataFrame([{L("Room / day", "कमरा/दिन"): inr(o.room_rent_per_day), L("Type", "प्रकार"): labels[o.label],
                                L("Insurer pays", "बीमा देगा"): inr(o.estimated_payout), L("You pay", "आप देंगे"): inr(o.you_pay)}
                               for o in options])
            st.dataframe(df, hide_index=True, width="stretch")
            mine = next((o for o in options if o.label == "Your room"), None)
            best = min(options, key=lambda o: o.you_pay)
            if mine and best.you_pay < mine.you_pay:
                st.success(L(f"Choosing a {inr(best.room_rent_per_day)}/day room would save you about {inr(mine.you_pay - best.you_pay)}.",
                             f"{inr(best.room_rent_per_day)}/दिन का कमरा लेने पर आपके लगभग {inr(mine.you_pay - best.you_pay)} बचेंगे।"))
        else:
            st.caption(L("Needs a room-rent limit in the policy and a room charge on the bill.",
                         "इसके लिए पॉलिसी में रूम-रेंट सीमा और बिल में कमरे का शुल्क होना चाहिए।"))

    with st.expander(L("📋 Documents you'll need for the claim", "📋 क्लेम के लिए ज़रूरी दस्तावेज़"), expanded=True):
        mode = st.radio(L("Claim type", "क्लेम का प्रकार"), ["cashless", "reimbursement"], horizontal=True, key="claim_mode",
                        format_func=lambda m: L("Cashless (network hospital)", "कैशलेस (नेटवर्क अस्पताल)") if m == "cashless"
                        else L("Reimbursement (pay first, claim later)", "प्रतिपूर्ति (पहले भुगतान, बाद में क्लेम)"))
        for i, doc in enumerate(claim_documents(mode, bill)):
            st.checkbox(bi(doc), key=f"cdoc_{mode}_{i}")
    if est.missing_information:
        note(L("Could not check: ", "जाँच नहीं हो सकी: ") + ", ".join(est.missing_information))
    with st.expander(L("Assumptions and policy clauses", "मान्यताएँ और पॉलिसी क्लॉज़")):
        for a in est.assumptions:
            st.markdown(f"- {a}")
        rel = run(get_claim_relevant_clauses, pid, {"diagnosis": bill.diagnosis, "procedure": bill.procedure})
        if rel and rel.clauses:
            for cl in rel.clauses[:5]:
                st.markdown(f"**{cl.condition_type.replace('_', ' ').title()}** · {L('Page', 'पेज')} {cl.page_number}\n\n> {cl.quoted_text}")
    st.download_button(L("⬇️ Download estimate summary", "⬇️ अनुमान सारांश डाउनलोड करें"), data=_summary_text(pid, est, bill),
                       file_name=f"claim_estimate_{date.today():%Y%m%d}.txt", mime="text/plain")
    st.caption(est.disclaimer)


def _summary_text(pid, est, bill) -> str:
    lines = [f"Paytm Saathi - Claim Pre-Flight estimate ({date.today():%d %b %Y})",
             f"Policy: {policy_label(pid)}", f"Hospital: {bill.hospital_name or '-'}",
             f"Diagnosis/procedure: {bill.diagnosis or '-'} / {bill.procedure or '-'}", "",
             f"Bill total:       Rs. {est.bill_total:,.2f}", f"Insurer may pay:  Rs. {est.estimated_payout:,.2f}",
             f"You may pay:      Rs. {est.patient_pays:,.2f}", f"Status: {est.eligibility_status}", "", "Steps:"]
    lines += [f"- {s.description}: -Rs. {s.deduction:,.2f} -> Rs. {s.amount_after:,.2f}" for s in est.steps]
    if est.flags:
        lines += ["", "Flags:"] + [f"- {f}" for f in est.flags]
    lines += ["", est.disclaimer]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Explain a clause
# ---------------------------------------------------------------------------
def tab_explain():
    mode = st.radio(L("How do you want to find the clause?", "क्लॉज़ कैसे ढूँढना है?"), ["paste", "number"], horizontal=True,
                    key="explain_mode", format_func=lambda m: L("Paste the wording", "शब्द पेस्ट करें") if m == "paste"
                    else L("By clause number", "क्लॉज़ नंबर से"))
    if mode == "number":
        if not need_policy():
            return
        ref = st.text_input(L("Clause number (for example 4.2)", "क्लॉज़ नंबर (जैसे 4.2)"), key="clause_ref")
        go = st.button(L("Explain", "समझाएँ"), type="primary", disabled=not ref.strip(), key="clause_go")
        res = run(explain_policy_clause, current_policy(), ref, lang()) if go else None
    else:
        text = st.text_area(L("Paste any confusing line from your policy", "पॉलिसी की कोई उलझन वाली लाइन पेस्ट करें"), height=130, key="clause_text",
                            placeholder=L("e.g. Expenses related to the treatment of cataract are payable after 24 months of continuous coverage…",
                                          "जैसे: मोतियाबिंद के इलाज से जुड़े खर्च 24 महीने के लगातार कवरेज के बाद देय हैं…"))
        go = st.button(L("Explain in simple words", "आसान शब्दों में समझाएँ"), type="primary", disabled=len(text.strip()) < 20, key="clause_text_go")
        res = run(explain_clause_text, text, lang()) if go else None
    if res:
        left, right = st.columns(2)
        left.markdown(f"**{L('Original wording', 'मूल शब्द')}**" + (f" · {L('page', 'पेज')} {res.page_number}" if res.page_number else ""))
        left.markdown(f'<div class="ps-msg">{esc(res.original_clause)}</div>', unsafe_allow_html=True)
        right.markdown(f"**{L('In simple words', 'आसान शब्दों में')}**")
        right.markdown(f'<div class="ps-note">{esc(res.simple_explanation)}</div>', unsafe_allow_html=True)
        if demo_mode():
            st.caption(L("Connect a free AI key for real explanations.", "असली व्याख्या के लिए मुफ़्त एआई कुंजी जोड़ें।"))


def tab_readiness():
    """Claim Pre-Flight + Cashless Hospital Check: Claim Readiness Report."""
    if not need_policy():
        return
    pid = current_policy()

    st.markdown(f"### {L('Claim Readiness Report', 'क्लेम तत्परता रिपोर्ट')}")
    st.caption(L("Check if your planned hospitalization is likely to be cashless and how much you may pay out of pocket.",
                 "जाँचें कि आपकी नियोजित अस्पताल में भर्ती कैशलेस होने की संभावना है और आपको अपनी जेब से कितना भुगतान करना पड़ सकता है।"))

    # -- Form for claim details -------------------------------------------
    with st.form("claim_readiness_form"):
        c1, c2 = st.columns(2)
        patient = c1.text_input(L("Patient / member name", "रोगी / सदस्य का नाम"), key="crf_patient")
        hospital = c2.text_input(L("Hospital name", "अस्पताल का नाम"), key="crf_hospital")
        c3, c4 = st.columns(2)
        city = c3.text_input(L("City", "शहर"), key="crf_city")
        treatment = c4.text_input(L("Planned treatment / procedure", "योजनाबद्ध इलाज / प्रक्रिया"), key="crf_treatment")
        c5, c6 = st.columns(2)
        room_type = c5.text_input(L("Room type (e.g. single, deluxe)", "कमरे का प्रकार (जैसे सिंगल, डीलक्स)"), key="crf_room")
        admit = c6.date_input(L("Expected admission date", "अपेक्षित भर्ती तिथि"), key="crf_admit")
        bill_amt = st.number_input(L("Estimated hospital bill (₹)", "अनुमानित अस्पताल बिल (₹)"), min_value=0.0, value=0.0, key="crf_bill")
        submitted = st.form_submit_button(L("Generate Claim Readiness Report", "क्लेम तत्परता रिपोर्ट बनाएँ"), type="primary", width="stretch")

    if not submitted:
        # Show sample questions
        st.markdown(f"#### {L('Try asking:', 'आज़माएँ:')}")
        cols = st.columns(3)
        samples = [
            ("Will my cataract surgery be cashless at Apollo Hospital Delhi?", "क्या मोतियाबिंद सर्जरी अपोलो अस्पताल दिल्ली में कैशलेस होगी?"),
            ("What documents do I need before admission?", "भर्ती से पहले मुझे कौन से दस्तावेज़ चाहिए?"),
            ("Why is my eligible amount lower than the hospital bill?", "मेरी पात्र राशि अस्पताल बिल से कम क्यों है?"),
        ]
        for i, (en, hi) in enumerate(samples):
            if cols[i % 3].button(L(en, hi), key=f"crf_sample_{i}", width="stretch"):
                st.session_state["crf_sample_q"] = L(en, hi)
        if "crf_sample_q" in st.session_state:
            st.info(st.session_state["crf_sample_q"])
        return

    # -- Generate report ---------------------------------------------------
    with st.spinner(L("Generating your Claim Readiness Report…", "आपकी क्लेम तत्परता रिपोर्ट बनाई जा रही है…")):
        report = run(generate_claim_readiness_report,
                     pid,
                     patient_name=patient or None,
                     hospital_name=hospital or None,
                     hospital_city=city or None,
                     treatment_procedure=treatment or None,
                     room_type=room_type or None,
                     expected_admission_date=admit.isoformat() if admit else None,
                     estimated_bill=bill_amt if bill_amt > 0 else None)

    if not report:
        return

    # Save the report
    try:
        save_claim_readiness_report(report)
    except Exception:
        pass  # non-fatal

    # -- Policy Card -------------------------------------------------------
    pc = report.policy_card
    st.markdown(f"#### {L('Policy Card', 'पॉलिसी कार्ड')}")
    m = st.columns(4)
    m[0].metric(L("Sum insured", "बीमित राशि"), inr(pc.sum_insured.value) if pc.sum_insured else "—")
    m[1].metric(L("Room rent limit", "कमरे के किराए की सीमा"), inr(pc.room_rent_limit.value) if pc.room_rent_limit else "—")
    m[2].metric(L("Co-pay", "सह-भुगतान"), f"{pc.copay_percentage:g}%" if pc.copay_percentage is not None else "—")
    m[3].metric(L("Deductible", "कटौती"), pc.deductibles or "—")
    if pc.insurer_name or pc.policy_name:
        st.caption(f"{pc.insurer_name or ''} · {pc.policy_name or ''}")

    # -- Readiness Score ---------------------------------------------------
    rs = report.readiness_score
    color_map = {"green": "ok", "amber": "warn", "red": "bad"}
    kind = color_map.get(rs.color.value, "info")
    st.markdown(f"#### {L('Claim Readiness Score', 'क्लेम तत्परता स्कोर')} &nbsp; {pill(f'{rs.score}/100 · {rs.label}', kind)}", unsafe_allow_html=True)
    st.progress(rs.score / 100.0, text=f"{rs.label_hi} · {rs.score}%")
    if rs.reasons:
        with st.expander(L("Why this score?", "यह स्कोर क्यों?"), expanded=True):
            for r, r_hi in zip(rs.reasons, rs.reasons_hi):
                st.markdown(f"- {L(r, r_hi)}")

    # -- Cashless Eligibility ---------------------------------------------
    ce = report.cashless_eligibility
    ce_kind = {"likely": "ok", "uncertain": "warn", "not_eligible": "bad"}.get(ce.status.value, "info")
    st.markdown(f"#### {L('Cashless Eligibility', 'कैशलेस पात्रता')} &nbsp; {pill(ce.label, ce_kind)}", unsafe_allow_html=True)
    if ce.reasons:
        for r, r_hi in zip(ce.reasons, ce.reasons_hi):
            st.markdown(f"- {L(r, r_hi)}")
    st.caption(L("Pre-authorization: 48 hours before planned admission; 24 hours for emergency.",
                 "प्री-ऑथराइज़ेशन: तय भर्ती से 48 घंटे पहले; इमरजेंसी में 24 घंटे के भीतर।"))

    # -- Pre-auth ----------------------------------------------------------
    pa = report.pre_auth
    with st.expander(L("Pre-authorization requirements", "प्री-ऑथराइज़ेशन आवश्यकताएँ"), expanded=False):
        st.markdown(f"**{pa.label}**")
        for s, s_hi in zip(pa.steps, pa.steps_hi):
            st.markdown(f"- {L(s, s_hi)}")
        if pa.timeline:
            st.caption(f"{L('Timeline', 'समय-सीमा')}: {L(pa.timeline, pa.timeline_hi)}")

    # -- Room rent impact --------------------------------------------------
    ri = report.room_rent_impact
    st.markdown(f"#### {L('Room rent impact', 'कमरे के किराए का असर')}")
    if ri.limit and ri.limit.value:
        st.markdown(f"{L(ri.impact_description, ri.impact_description_hi)}")
        if ri.proportionate_deduction_warning:
            alert(L("A costlier room may reduce doctor, surgery and nursing payouts proportionately.",
                    "महँगा कमरा लेने पर डॉक्टर, सर्जरी और नर्सिंग का भुगतान आनुपातिक रूप से घट सकता है।"))
    else:
        st.info(L(ri.impact_description, ri.impact_description_hi))

    # -- Financial estimate ------------------------------------------------
    fe = report.financial_estimate
    st.markdown(f"#### {L('Estimated out-of-pocket exposure', 'अनुमानित जेब से खर्च')}")
    m = st.columns(3)
    m[0].metric(L("Estimated bill", "अनुमानित बिल"), inr(fe.bill_total))
    m[1].metric(L("Insurer may pay", "बीमा कंपनी दे सकती है"), inr(fe.estimated_payout))
    m[2].metric(L("You may pay", "आपको देना पड़ सकता है"), inr(fe.patient_pays))
    if fe.out_of_pocket_range_low > 0:
        st.caption(f"{L('Estimated range', 'अनुमानित दायरा')}: {inr(fe.out_of_pocket_range_low)} – {inr(fe.out_of_pocket_range_high)}")
    st.caption(L("Informational estimate only — not an insurer approval.",
                 "केवल सूचनात्मक अनुमान — बीमा कंपनी की मंज़ूरी नहीं।"))

    # -- Waiting periods & exclusions --------------------------------------
    if report.waiting_periods_relevant or report.exclusions_relevant:
        with st.expander(L("Waiting periods & exclusions", "प्रतीक्षा अवधि और अपवर्जन"), expanded=False):
            if report.waiting_periods_relevant:
                st.markdown(f"**{L('Waiting periods', 'प्रतीक्षा अवधियाँ')}**")
                for w in report.waiting_periods_relevant:
                    st.markdown(f"- {w.condition}: {w.duration or f'{w.duration_days} days'} "
                                f"(p.{w.source_page or '-'})")
            if report.exclusions_relevant:
                st.markdown(f"**{L('Exclusions', 'अपवर्जन')}**")
                for e in report.exclusions_relevant:
                    st.markdown(f"- {e.condition} (p.{e.source_page or '-'})")

    # -- Documents ---------------------------------------------------------
    st.markdown(f"#### {L('Required documents', 'ज़रूरी दस्तावेज़')}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{L('Cashless', 'कैशलेस')}**")
        for d in report.required_documents_cashless:
            st.checkbox(d, key=f"crf_doc_cashless_{hash(d)}")
    with c2:
        st.markdown(f"**{L('Reimbursement', 'प्रतिपूर्ति')}**")
        for d in report.required_documents_reimbursement:
            st.checkbox(d, key=f"crf_doc_reimb_{hash(d)}")

    # -- Citations ---------------------------------------------------------
    if report.citations:
        with st.expander(L("Policy citations (source text)", "पॉलिसी उद्धरण (मूल पाठ)"), expanded=False):
            for cit in report.citations:
                head = f"📄 {L('Page', 'पेज')} {cit.page_number}" + (f" · {L('Clause', 'क्लॉज़')} {cit.clause_reference}" if cit.clause_reference else "")
                st.markdown(f'<div class="ps-note"><b>{esc(head)}</b><br>{esc(cit.quoted_text)}</div>', unsafe_allow_html=True)

    # -- Download ----------------------------------------------------------
    import json
    report_json = report.model_dump_json(indent=2)
    st.download_button(
        L("⬇️ Download Claim Readiness Report (JSON)", "⬇️ क्लेम तत्परता रिपोर्ट डाउनलोड करें (JSON)"),
        data=report_json,
        file_name=f"claim_readiness_{report.report_id}.json",
        mime="application/json",
        width="stretch",
    )

    st.divider()
    st.caption(report.disclaimer)


def render():
    page_header(L("Health insurance", "हेल्थ बीमा"),
                L("Understand your cover, check claims before you pay, and decode the fine print.",
                  "अपना कवर समझें, भुगतान से पहले क्लेम जाँचें और बारीक शर्तें समझें।"))
    tabs = st.tabs([L("📄 My policy", "📄 मेरी पॉलिसी"), L("💬 Ask Saathi", "💬 साथी से पूछें"),
                    L("🧾 Claim Pre-Flight", "🧾 क्लेम प्री-फ्लाइट"),
                    L("🏥 Claim Readiness", "🏥 क्लेम तत्परता"),
                    L("🔍 Explain a clause", "🔍 क्लॉज़ समझें")])
    for tab, fn in zip(tabs, (tab_policy, tab_ask, tab_claim, tab_readiness, tab_explain)):
        with tab:
            fn()