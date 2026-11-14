"""Policy insights built on the extracted PolicyDetails - deterministic, no LLM.

* coverage_at_a_glance - Covered / Not covered / Watch-outs, each with clause + page
* waiting_period_timeline - when each waiting period ends, from the policy start date
* room_rent_what_if - payout for different room choices (the #1 claim surprise)
* claim_documents - checklist for cashless vs reimbursement claims
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from pydantic import BaseModel, Field

from .exceptions import CalculationInputError
from .payout_calculator import _limit_value, estimate_payout
from .schemas import BillLineItem, ClaimContext, HospitalBill, PolicyDetails


class Insight(BaseModel):
    text: str
    text_hi: str
    clause: Optional[str] = None
    page: Optional[int] = None


class CoverageGlance(BaseModel):
    covered: list[Insight] = Field(default_factory=list)
    not_covered: list[Insight] = Field(default_factory=list)
    watch_outs: list[Insight] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)


def _rupees(value: float) -> str:
    return f"₹{value:,.0f}"


def coverage_at_a_glance(details: PolicyDetails) -> CoverageGlance:
    g = CoverageGlance(unknown=list(details.unknown_fields))
    si = details.sum_insured.value if details.sum_insured else None
    if si:
        s = details.sum_insured
        g.covered.append(Insight(text=f"Hospital bills up to {_rupees(si)} a year (sum insured)",
                                 text_hi=f"साल में {_rupees(si)} तक के अस्पताल खर्च (बीमित राशि)",
                                 clause=s.source_clause, page=s.source_page))
    if details.pre_hospitalization_coverage_period:
        g.covered.append(Insight(text=f"Costs before admission: {details.pre_hospitalization_coverage_period}",
                                 text_hi=f"भर्ती से पहले के खर्च: {details.pre_hospitalization_coverage_period}"))
    if details.post_hospitalization_coverage_period:
        g.covered.append(Insight(text=f"Costs after discharge: {details.post_hospitalization_coverage_period}",
                                 text_hi=f"छुट्टी के बाद के खर्च: {details.post_hospitalization_coverage_period}"))
    if details.network_hospital_conditions:
        g.covered.append(Insight(text=f"Cashless at network hospitals: {details.network_hospital_conditions[:140]}",
                                 text_hi="नेटवर्क अस्पतालों में कैशलेस इलाज (शर्तें लागू)"))
    for e in details.exclusions[:8]:
        g.not_covered.append(Insight(text=e.condition, text_hi=e.condition, clause=e.source_clause, page=e.source_page))

    room = _limit_value(details.room_rent_limit, si)
    if room:
        r = details.room_rent_limit
        g.watch_outs.append(Insight(
            text=f"Room rent is capped at {_rupees(room)}/day. A costlier room also cuts doctor and surgery payouts.",
            text_hi=f"कमरे का किराया {_rupees(room)}/दिन तक सीमित है। महँगा कमरा लेने पर डॉक्टर और सर्जरी का भुगतान भी घटता है।",
            clause=r.source_clause, page=r.source_page))
    icu = _limit_value(details.icu_room_rent_limit, si)
    if icu:
        i = details.icu_room_rent_limit
        g.watch_outs.append(Insight(text=f"ICU charges are capped at {_rupees(icu)}/day.",
                                    text_hi=f"आईसीयू शुल्क {_rupees(icu)}/दिन तक सीमित है।",
                                    clause=i.source_clause, page=i.source_page))
    if details.copay_percentage:
        g.watch_outs.append(Insight(text=f"You pay {details.copay_percentage:g}% of every claim yourself (co-pay).",
                                    text_hi=f"हर क्लेम का {details.copay_percentage:g}% आपको खुद देना होगा (सह-भुगतान)।"))
    if details.deductibles:
        g.watch_outs.append(Insight(text=f"Deductible: {details.deductibles}", text_hi=f"कटौती: {details.deductibles}"))
    for w in _all_waits(details):
        if w.duration_days or w.duration:
            g.watch_outs.append(Insight(
                text=f"Waiting period for {w.condition}: {w.duration or str(w.duration_days) + ' days'}",
                text_hi=f"{w.condition} के लिए प्रतीक्षा अवधि: {w.duration or str(w.duration_days) + ' दिन'}",
                clause=w.source_clause, page=w.source_page))
    for s in details.sub_limits[:6]:
        g.watch_outs.append(Insight(text=f"Sub-limit for {s.condition}: {s.limit or 'see policy'}",
                                    text_hi=f"{s.condition} की उप-सीमा: {s.limit or 'पॉलिसी देखें'}",
                                    clause=s.source_clause, page=s.source_page))
    return g


def _all_waits(details: PolicyDetails):
    seen, out = set(), []
    for w in ([details.initial_waiting_period, details.pre_existing_disease_waiting_period]
              + list(details.waiting_periods) + list(details.specific_disease_waiting_periods)):
        if w is None:
            continue
        key = (w.condition.lower(), w.duration_days)
        if key not in seen:
            seen.add(key)
            out.append(w)
    return out


class WaitingRow(BaseModel):
    condition: str
    duration_days: int
    ends_on: str
    days_left: int
    progress: float  # 0..1
    completed: bool
    clause: Optional[str] = None
    page: Optional[int] = None


def waiting_period_timeline(details: PolicyDetails, policy_start: date,
                            today: Optional[date] = None) -> list[WaitingRow]:
    today = today or date.today()
    if policy_start > today:
        raise CalculationInputError("Policy start date cannot be in the future.")
    elapsed = (today - policy_start).days
    rows = []
    for w in _all_waits(details):
        if not w.duration_days:
            continue
        ends = policy_start + timedelta(days=w.duration_days)
        rows.append(WaitingRow(
            condition=w.condition, duration_days=w.duration_days, ends_on=ends.isoformat(),
            days_left=max((ends - today).days, 0), progress=min(elapsed / w.duration_days, 1.0),
            completed=elapsed >= w.duration_days, clause=w.source_clause, page=w.source_page))
    return sorted(rows, key=lambda r: r.days_left)


class RoomOption(BaseModel):
    label: str
    room_rent_per_day: float
    estimated_payout: float
    you_pay: float
    within_limit: Optional[bool] = None


def room_rent_what_if(details: PolicyDetails, bill: HospitalBill, context: Optional[ClaimContext] = None,
                      rates: Optional[list[float]] = None) -> list[RoomOption]:
    """Payout for the same treatment with different room rents (keeps every other charge)."""
    days = bill.days_of_stay or int(sum(i.quantity or 0 for i in bill.line_items if i.category == "room_rent"))
    if not days:
        raise CalculationInputError("Enter the number of days in hospital to compare room choices.")
    others = [i for i in bill.line_items if i.category != "room_rent"]
    current_room = sum(i.amount for i in bill.line_items if i.category == "room_rent" and i.is_payable)
    limit = _limit_value(details.room_rent_limit, details.sum_insured.value if details.sum_insured else None)
    if rates is None:
        rates = sorted({round(current_room / days)} | ({round(limit), round(limit * 1.5), round(limit * 2)} if limit else set()))
    options = []
    for rate in rates:
        if rate <= 0:
            continue
        items = others + [BillLineItem(description="Room rent", category="room_rent",
                                       quantity=days, unit_price=rate, amount=rate * days)]
        est = estimate_payout(details, bill.model_copy(update={"line_items": items, "days_of_stay": days}), context)
        is_current = abs(rate * days - current_room) < 1
        label = "Your room" if is_current else ("Within limit" if limit and rate <= limit else "Costlier room")
        options.append(RoomOption(label=label, room_rent_per_day=rate, estimated_payout=est.estimated_payout,
                                  you_pay=est.patient_pays, within_limit=(rate <= limit) if limit else None))
    return options


def claim_documents(mode: str, bill: Optional[HospitalBill] = None) -> list[dict]:
    """Checklist for a 'cashless' or 'reimbursement' claim, tailored to the bill."""
    if mode not in ("cashless", "reimbursement"):
        raise CalculationInputError("mode must be 'cashless' or 'reimbursement'.")
    docs = [
        {"en": "Health card / policy number and photo ID", "hi": "हेल्थ कार्ड / पॉलिसी नंबर और फोटो आईडी"},
        {"en": "Doctor's admission advice / prescription", "hi": "डॉक्टर की भर्ती सलाह / पर्चा"},
    ]
    if mode == "cashless":
        docs += [
            {"en": "Pre-authorisation form, filled at the hospital's insurance (TPA) desk", "hi": "अस्पताल के बीमा (TPA) डेस्क पर भरा प्री-ऑथराइज़ेशन फॉर्म"},
            {"en": "Inform the insurer: 48 hours before a planned admission, within 24 hours of an emergency", "hi": "बीमा कंपनी को सूचना: तय भर्ती से 48 घंटे पहले, इमरजेंसी में 24 घंटे के भीतर"},
            {"en": "At discharge: sign the final bill and keep a copy of the discharge summary", "hi": "छुट्टी पर: फाइनल बिल पर हस्ताक्षर करें और डिस्चार्ज समरी की कॉपी रखें"},
        ]
    else:
        docs += [
            {"en": "Claim form signed by you", "hi": "आपके हस्ताक्षर वाला क्लेम फॉर्म"},
            {"en": "Original final hospital bill with itemised breakup and payment receipts", "hi": "मूल फाइनल बिल (आइटम-वार) और भुगतान की रसीदें"},
            {"en": "Discharge summary", "hi": "डिस्चार्ज समरी"},
            {"en": "All test reports and pharmacy bills with prescriptions", "hi": "सभी जाँच रिपोर्ट और दवा के बिल पर्चे सहित"},
            {"en": "Cancelled cheque or bank details for the payout", "hi": "भुगतान के लिए कैंसल चेक या बैंक विवरण"},
            {"en": "Submit within the policy deadline (often 15-30 days after discharge)", "hi": "पॉलिसी की समय-सीमा में जमा करें (अक्सर छुट्टी के 15-30 दिन में)"},
        ]
    if bill is not None:
        cats = {i.category for i in bill.line_items}
        if "surgery" in cats:
            docs.append({"en": "Operation theatre notes and implant invoice/sticker (if any)", "hi": "ऑपरेशन थिएटर नोट्स और इम्प्लांट का बिल/स्टिकर (यदि हो)"})
        if "icu_rent" in cats:
            docs.append({"en": "ICU admission notes", "hi": "आईसीयू भर्ती नोट्स"})
        text = f"{bill.diagnosis or ''} {bill.procedure or ''}".lower()
        if any(k in text for k in ("accident", "fracture", "injury", "rta")):
            docs.append({"en": "FIR or Medico-Legal Certificate (accident cases)", "hi": "एफआईआर या मेडिको-लीगल सर्टिफिकेट (दुर्घटना के मामले)"})
    return docs
