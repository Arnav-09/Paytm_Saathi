"""Claim payout estimation - deterministic, transparent, no LLM arithmetic.

Takes the structured :class:`PolicyDetails` (extracted from the policy) and a
:class:`HospitalBill`, and walks through the usual Indian health-insurance
deductions in order, recording every step with the policy clause it relies on:

1. Bill total
2. Non-payable items removed
3. Room rent cap (+ proportionate deduction on room-linked charges)
4. ICU rent cap
5. Treatment sub-limit
6. Deductible
7. Co-pay
8. Remaining sum insured cap

It also checks waiting periods and exclusions. The result is an *estimate*:
conventions differ between insurers (order of co-pay vs deductible, which charges
are proportionately reduced), so every assumption is listed in the output.
"""

from __future__ import annotations

import re
from typing import Optional

from .exceptions import CalculationInputError
from .schemas import (
    ClaimContext,
    HospitalBill,
    MoneyValue,
    PayoutEstimate,
    PayoutStep,
    PolicyDetails,
)

# Charges that are usually reduced in proportion when the room is above the limit.
PROPORTIONATE_CATEGORIES = {"doctor_fees", "surgery", "nursing", "other"}

_GENERIC_WORDS = {
    "treatment", "treatments", "expenses", "expense", "surgery", "surgeries", "disease",
    "diseases", "hospitalization", "hospitalisation", "related", "charges", "medical",
    "condition", "conditions", "policy", "cover", "covered", "claim", "claims", "waiting",
    "period", "periods", "existing", "within", "which", "arising", "from", "with", "that",
    "this", "than", "other", "following", "including", "excluding", "after", "before",
    "years", "year", "days", "month", "months", "benefit", "benefits", "illness", "injury",
    "procedure", "procedures", "pre-existing", "first", "only", "under",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-]{3,}", (text or "").lower())
    return {w for w in words if w not in _GENERIC_WORDS}


def _matches(condition: str, claim_text: str) -> bool:
    """True when a policy condition shares a specific medical keyword with the claim."""
    cond = _keywords(condition)
    claim = _keywords(claim_text)
    for c in cond:
        for k in claim:
            if c == k or (len(c) >= 6 and len(k) >= 6 and (c[:6] == k[:6])):
                return True
    return False


def parse_amount(text: Optional[str], sum_insured: Optional[float] = None) -> Optional[float]:
    """Parse '₹50,000', 'Rs. 2 lakh', '1% of sum insured' into rupees."""
    if not text:
        return None
    lowered = text.lower()
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(?:the\s*)?sum\s*insured", lowered)
    if pct:
        return round(float(pct.group(1)) * sum_insured / 100.0, 2) if sum_insured else None
    scale = re.search(r"(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)", lowered)
    if scale:
        factor = 10_000_000 if scale.group(2).startswith("crore") else 100_000
        return round(float(scale.group(1)) * factor, 2)
    money = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)", lowered)
    if money:
        return float(money.group(1).replace(",", ""))
    plain = re.search(r"\b(\d{1,3}(?:,\d{2,3})+|\d{4,})(?:\.\d+)?\b", lowered)
    if plain:
        return float(plain.group(1).replace(",", ""))
    return None


def _limit_value(money: Optional[MoneyValue], sum_insured: Optional[float]) -> Optional[float]:
    """Resolve a limit that may be stated as a % of the sum insured."""
    if money is None:
        return None
    if money.quoted_text and re.search(r"%\s*of\s*(?:the\s*)?sum\s*insured", money.quoted_text, re.IGNORECASE):
        resolved = parse_amount(money.quoted_text, sum_insured)
        if resolved:
            return resolved
    return money.value


def _r(value: float) -> float:
    return round(value + 0.0, 2)


def estimate_payout(
    details: PolicyDetails,
    bill: HospitalBill,
    context: Optional[ClaimContext] = None,
    policy_id: Optional[str] = None,
) -> PayoutEstimate:
    """Estimate the insurer payout for a bill under a policy. See module docstring."""
    context = context or ClaimContext()
    if not bill.line_items:
        raise CalculationInputError("The bill has no line items to calculate on.")
    if any(item.amount < 0 for item in bill.line_items):
        raise CalculationInputError("Bill amounts cannot be negative.")
    if context.previously_claimed < 0:
        raise CalculationInputError("previously_claimed cannot be negative.")

    items = bill.line_items
    sum_insured = details.sum_insured.value if details.sum_insured else None
    est = PayoutEstimate(policy_id=policy_id, bill_total=_r(sum(i.amount for i in items)))
    flags_en: list[str] = []
    flags_hi: list[str] = []

    def flag(en: str, hi: str) -> None:
        flags_en.append(en)
        flags_hi.append(hi)

    def step(name, en, hi, before, after, clause=None, page=None):
        est.steps.append(
            PayoutStep(
                step=name, description=en, description_hi=hi,
                amount_before=_r(before), amount_after=_r(after),
                deduction=_r(before - after), clause_reference=clause, source_page=page,
            )
        )

    running = est.bill_total
    step("bill_total", "Total amount on the hospital bill", "अस्पताल के बिल की कुल राशि", running, running)

    # 1. Non-payable items
    non_payable = sum(i.amount for i in items if not i.is_payable)
    est.non_payable_total = _r(non_payable)
    if non_payable > 0:
        step("non_payable", "Items that insurers do not pay (registration, attendant charges etc.) removed",
             "जिन खर्चों का बीमा कंपनी भुगतान नहीं करती (रजिस्ट्रेशन, अटेंडेंट आदि) हटाए गए",
             running, running - non_payable)
        running -= non_payable

    # 2. Room rent cap + proportionate deduction
    room_amount = sum(i.amount for i in items if i.category == "room_rent" and i.is_payable)
    room_limit = _limit_value(details.room_rent_limit, sum_insured)
    days = bill.days_of_stay
    if days is None:
        qty = sum(i.quantity or 0 for i in items if i.category == "room_rent")
        days = int(qty) if qty else None
    rr = details.room_rent_limit
    if room_amount > 0:
        if room_limit is None:
            est.missing_information.append("room_rent_limit")
            est.assumptions.append("Room rent limit not found in the policy, so no room rent cap was applied.")
        elif not days:
            est.missing_information.append("days_of_stay")
            flag("Number of days in hospital is unknown, so the room rent limit could not be checked.",
                 "अस्पताल में रहने के दिन पता नहीं हैं, इसलिए कमरे के किराए की सीमा जाँची नहीं जा सकी।")
        else:
            per_day = room_amount / days
            if per_day > room_limit:
                allowed = room_limit * days
                room_ded = room_amount - allowed
                step("room_rent_cap",
                     f"Room rent ₹{per_day:,.0f}/day is above your limit of ₹{room_limit:,.0f}/day; excess removed",
                     f"कमरे का किराया ₹{per_day:,.0f}/दिन आपकी सीमा ₹{room_limit:,.0f}/दिन से ज़्यादा है; अतिरिक्त राशि हटाई गई",
                     running, running - room_ded,
                     rr.source_clause if rr else None, rr.source_page if rr else None)
                running -= room_ded
                est.room_rent_deduction = _r(room_ded)
                if context.apply_proportionate_deduction:
                    factor = room_limit / per_day
                    base = sum(i.amount for i in items if i.is_payable and i.category in PROPORTIONATE_CATEGORIES)
                    prop = base * (1 - factor)
                    if prop > 0:
                        step("proportionate_deduction",
                             f"Doctor, surgery and nursing charges reduced to {factor:.0%} because the room was costlier than the limit",
                             f"कमरा सीमा से महंगा होने के कारण डॉक्टर, सर्जरी और नर्सिंग खर्च घटाकर {factor:.0%} किए गए",
                             running, running - prop,
                             rr.source_clause if rr else None, rr.source_page if rr else None)
                        running -= prop
                        est.proportionate_deduction = _r(prop)
                        est.assumptions.append(
                            "Proportionate deduction applied to doctor, surgery, nursing and other charges "
                            "(not medicines, tests or consumables). Check your policy wording.")
                flag("You chose a room costlier than your policy allows; this reduces the payout.",
                     "आपने पॉलिसी की अनुमति से महँगा कमरा लिया; इससे भुगतान कम होगा।")
    if details.room_rent_limit is not None and room_limit is not None:
        est.assumptions.append("Room rent limit is treated as a per-day limit." if context.room_rent_is_per_day
                               else "Room rent limit is treated as a per-stay limit.")

    # 3. ICU cap
    icu_amount = sum(i.amount for i in items if i.category == "icu_rent" and i.is_payable)
    icu_limit = _limit_value(details.icu_room_rent_limit, sum_insured)
    if icu_amount > 0 and icu_limit is not None:
        icu_days = sum(i.quantity or 0 for i in items if i.category == "icu_rent") or None
        if icu_days:
            per_day = icu_amount / icu_days
            if per_day > icu_limit:
                ded = icu_amount - icu_limit * icu_days
                ic = details.icu_room_rent_limit
                step("icu_cap", f"ICU charges ₹{per_day:,.0f}/day are above the limit of ₹{icu_limit:,.0f}/day",
                     f"आईसीयू शुल्क ₹{per_day:,.0f}/दिन सीमा ₹{icu_limit:,.0f}/दिन से ज़्यादा है",
                     running, running - ded, ic.source_clause if ic else None, ic.source_page if ic else None)
                running -= ded
                est.room_rent_deduction = _r(est.room_rent_deduction + ded)
        else:
            est.missing_information.append("icu_days")
    elif icu_amount > 0:
        est.missing_information.append("icu_room_rent_limit")

    # 4. Treatment sub-limit
    claim_text = " ".join(filter(None, [bill.diagnosis, bill.procedure]))
    caps = []
    for sub in details.sub_limits:
        if claim_text and _matches(sub.condition, claim_text):
            cap = parse_amount(sub.limit, sum_insured)
            if cap is not None:
                caps.append((cap, sub))
    if caps:
        cap, sub = min(caps, key=lambda pair: pair[0])
        if running > cap:
            step("sub_limit", f"Sub-limit for '{sub.condition}' caps the claim at ₹{cap:,.0f}",
                 f"'{sub.condition}' की उप-सीमा के कारण क्लेम ₹{cap:,.0f} तक सीमित",
                 running, cap, sub.source_clause, sub.source_page)
            est.sub_limit_deduction = _r(running - cap)
            running = cap
            est.assumptions.append("Sub-limit applied to the whole admissible bill; some policies apply it only to the procedure charges.")
    elif details.sub_limits and not claim_text:
        est.missing_information.append("diagnosis_or_procedure")

    # 5. Deductible
    deductible = parse_amount(details.deductibles, sum_insured)
    if deductible:
        applied = min(deductible, running)
        step("deductible", f"Policy deductible of ₹{deductible:,.0f} is paid by you first",
             f"पॉलिसी की ₹{deductible:,.0f} की कटौती पहले आपको देनी होगी", running, running - applied)
        running -= applied
        est.deductible_applied = _r(applied)

    # 6. Co-pay
    if details.copay_percentage:
        copay = running * details.copay_percentage / 100.0
        step("copay", f"Co-pay: you share {details.copay_percentage:g}% of the admissible amount",
             f"सह-भुगतान: आप स्वीकार्य राशि का {details.copay_percentage:g}% देंगे", running, running - copay)
        running -= copay
        est.copay_amount = _r(copay)
        est.assumptions.append("Co-pay is applied after the deductible.")

    # 7. Sum insured cap
    if sum_insured is None:
        est.missing_information.append("sum_insured")
    else:
        available = max(sum_insured - context.previously_claimed, 0.0)
        if running > available:
            si = details.sum_insured
            step("sum_insured_cap", f"Only ₹{available:,.0f} of your sum insured remains available",
                 f"आपकी बीमित राशि में केवल ₹{available:,.0f} बची है", running, available,
                 si.source_clause if si else None, si.source_page if si else None)
            est.sum_insured_cap_deduction = _r(running - available)
            running = available

    est.payout_if_covered = _r(max(running, 0.0))

    # 8. Eligibility checks
    status = "likely_payable"
    for exclusion in details.exclusions:
        if claim_text and _matches(exclusion.condition, claim_text):
            status = "needs_review"
            flag(f"The treatment may fall under a policy exclusion: \"{exclusion.condition}\""
                 + (f" (clause {exclusion.source_clause})" if exclusion.source_clause else "") + ".",
                 f"यह इलाज पॉलिसी के अपवर्जन में आ सकता है: \"{exclusion.condition}\"।")
            break
    waits = list(details.waiting_periods) + list(details.specific_disease_waiting_periods)
    if context.is_pre_existing and details.pre_existing_disease_waiting_period:
        waits.append(details.pre_existing_disease_waiting_period)
    for wait in waits:
        applies = (claim_text and _matches(wait.condition, claim_text)) or (
            context.is_pre_existing and "pre-exist" in wait.condition.lower())
        if not applies or not wait.duration_days:
            continue
        if context.policy_age_days is None:
            status = status if status == "likely_excluded" else "needs_review"
            est.missing_information.append("policy_age_days")
            flag(f"A waiting period of {wait.duration_days} days applies to '{wait.condition}'. Enter how old your policy is to check this.",
                 f"'{wait.condition}' पर {wait.duration_days} दिन की प्रतीक्षा अवधि लागू है। जाँच के लिए पॉलिसी की उम्र दर्ज करें।")
        elif context.policy_age_days < wait.duration_days:
            status = "likely_excluded"
            flag(f"Waiting period not completed for '{wait.condition}': policy is {context.policy_age_days} days old, "
                 f"{wait.duration_days} days required.",
                 f"'{wait.condition}' की प्रतीक्षा अवधि पूरी नहीं हुई: पॉलिसी {context.policy_age_days} दिन पुरानी है, {wait.duration_days} दिन ज़रूरी।")
    initial = details.initial_waiting_period
    if initial and initial.duration_days and context.policy_age_days is not None \
            and context.policy_age_days < initial.duration_days and status == "likely_payable":
        status = "needs_review"
        flag(f"Initial waiting period of {initial.duration_days} days may apply (accidents are usually exempt).",
             f"{initial.duration_days} दिन की शुरुआती प्रतीक्षा अवधि लागू हो सकती है (दुर्घटना आमतौर पर छूट में)।")

    if bill.stated_total and abs(bill.stated_total - est.bill_total) > max(1.0, 0.02 * bill.stated_total):
        flag("Bill line items do not add up to the stated bill total. Check for missing items.",
             "बिल के आइटम कुल राशि से मेल नहीं खाते। छूटे हुए आइटम जाँचें।")

    est.eligibility_status = status
    est.estimated_payout = 0.0 if status == "likely_excluded" else est.payout_if_covered
    est.patient_pays = _r(est.bill_total - est.estimated_payout)
    est.flags, est.flags_hi = flags_en, flags_hi
    est.missing_information = sorted(set(est.missing_information))
    est.assumptions = list(dict.fromkeys(est.assumptions))
    return est
