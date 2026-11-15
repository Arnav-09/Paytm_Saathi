"""Lending journeys: EMI, eligibility, documents, offer comparison, plain-language terms.

All numbers come from deterministic code. The eligibility rules and interest-rate
bands below are *illustrative defaults* (editable in ``LENDING_RULES``) - they are not
any lender's actual policy, and every result says so. The LLM is only used to explain
agreement text in plain language, never to decide eligibility or compute money.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import CalculationInputError, LLMProviderError
from .language import translate_text
from .llm_provider import LLMProvider, MockLLMProvider

LOAN_TYPES = ["personal", "home", "vehicle", "business", "education", "gold"]

# Illustrative rules. rate_bands: (min_score, annual_rate_pct) high->low score.
LENDING_RULES: dict[str, dict] = {
    "personal": {"max_foir": 0.50, "max_tenure_m": 60, "min_income": 15000, "min_score": 650,
                 "rate_bands": [(750, 11.5), (700, 14.0), (650, 18.0)], "secured": False},
    "home": {"max_foir": 0.55, "max_tenure_m": 360, "min_income": 20000, "min_score": 650,
             "rate_bands": [(750, 8.75), (700, 9.5), (650, 10.5)], "secured": True},
    "vehicle": {"max_foir": 0.50, "max_tenure_m": 84, "min_income": 15000, "min_score": 650,
                "rate_bands": [(750, 9.5), (700, 11.0), (650, 13.5)], "secured": True},
    "business": {"max_foir": 0.45, "max_tenure_m": 84, "min_income": 25000, "min_score": 680,
                 "rate_bands": [(750, 13.0), (700, 16.0), (680, 20.0)], "secured": False},
    "education": {"max_foir": 0.55, "max_tenure_m": 180, "min_income": 0, "min_score": 0,
                  "rate_bands": [(750, 9.5), (700, 10.5), (0, 12.0)], "secured": True},
    "gold": {"max_foir": 0.60, "max_tenure_m": 36, "min_income": 0, "min_score": 0,
             "rate_bands": [(0, 9.5)], "secured": True},
}
RETIREMENT_AGE = {"salaried": 60, "self_employed": 65}
DISCLAIMER = (
    "Indicative result using illustrative rules. It is not a loan offer or approval; "
    "the lender decides eligibility and the final interest rate."
)


# ---------------------------------------------------------------------------
# EMI maths
# ---------------------------------------------------------------------------
def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    if principal <= 0:
        raise CalculationInputError("Loan amount must be greater than zero.")
    if tenure_months <= 0:
        raise CalculationInputError("Tenure must be at least 1 month.")
    if annual_rate_pct < 0 or annual_rate_pct > 100:
        raise CalculationInputError("Interest rate must be between 0 and 100%.")
    r = annual_rate_pct / 12.0 / 100.0
    if r == 0:
        return round(principal / tenure_months, 2)
    factor = (1 + r) ** tenure_months
    return round(principal * r * factor / (factor - 1), 2)


def max_principal_for_emi(emi: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Largest loan whose EMI does not exceed ``emi``."""
    if emi <= 0 or tenure_months <= 0:
        return 0.0
    r = annual_rate_pct / 12.0 / 100.0
    if r == 0:
        return round(emi * tenure_months, 2)
    return round(emi * ((1 + r) ** tenure_months - 1) / (r * (1 + r) ** tenure_months), 2)


class ScheduleRow(BaseModel):
    month: int
    emi: float
    interest: float
    principal: float
    balance: float


class LoanSummary(BaseModel):
    principal: float
    annual_rate_pct: float
    tenure_months: int
    emi: float
    total_interest: float
    total_payment: float
    interest_share_pct: float


def amortization_schedule(principal: float, annual_rate_pct: float, tenure_months: int) -> list[ScheduleRow]:
    emi = calculate_emi(principal, annual_rate_pct, tenure_months)
    r = annual_rate_pct / 12.0 / 100.0
    balance, rows = principal, []
    for month in range(1, tenure_months + 1):
        interest = balance * r
        principal_part = min(emi - interest, balance)
        if month == tenure_months:  # absorb rounding in the final instalment
            principal_part = balance
        balance = max(balance - principal_part, 0.0)
        rows.append(ScheduleRow(month=month, emi=round(principal_part + interest, 2),
                                interest=round(interest, 2), principal=round(principal_part, 2),
                                balance=round(balance, 2)))
    return rows


def loan_summary(principal: float, annual_rate_pct: float, tenure_months: int) -> LoanSummary:
    rows = amortization_schedule(principal, annual_rate_pct, tenure_months)
    total_payment = round(sum(row.emi for row in rows), 2)
    total_interest = round(total_payment - principal, 2)
    return LoanSummary(
        principal=principal, annual_rate_pct=annual_rate_pct, tenure_months=tenure_months,
        emi=calculate_emi(principal, annual_rate_pct, tenure_months),
        total_interest=total_interest, total_payment=total_payment,
        interest_share_pct=round(100 * total_interest / total_payment, 1) if total_payment else 0.0,
    )


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
class LoanApplicant(BaseModel):
    model_config = ConfigDict(extra="ignore")

    loan_type: str = "personal"
    employment_type: str = "salaried"  # salaried | self_employed
    age: int = 30
    monthly_income: float
    existing_emi: float = 0.0
    credit_score: Optional[int] = None  # 300-900; None = unknown
    requested_amount: float
    tenure_months: int


class EligibilityResult(BaseModel):
    decision: str  # likely_eligible | partially_eligible | unlikely
    loan_type: str
    indicative_rate_pct: Optional[float] = None
    max_eligible_amount: float = 0.0
    requested_amount: float
    emi_for_requested: Optional[float] = None
    max_affordable_emi: float = 0.0
    foir_after_loan_pct: Optional[float] = None
    suggested_tenure_months: Optional[int] = None
    reasons: list[str] = Field(default_factory=list)
    reasons_hi: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)
    tips_hi: list[str] = Field(default_factory=list)
    documents: list[dict] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


def _rate_for(rules: dict, score: Optional[int]) -> Optional[float]:
    bands = rules["rate_bands"]
    if score is None:
        return bands[-1][1]  # unknown score: assume the most conservative band
    for min_score, rate in bands:
        if score >= min_score:
            return rate
    return None


def check_loan_eligibility(applicant: LoanApplicant) -> EligibilityResult:
    a = applicant
    if a.loan_type not in LENDING_RULES:
        raise CalculationInputError(f"Unsupported loan type. Choose one of: {', '.join(LOAN_TYPES)}.")
    if a.monthly_income < 0 or a.existing_emi < 0 or a.requested_amount <= 0 or a.tenure_months <= 0:
        raise CalculationInputError("Income, EMIs, amount and tenure must be positive numbers.")
    if not 18 <= a.age <= 80:
        raise CalculationInputError("Age must be between 18 and 80.")
    if a.credit_score is not None and not 300 <= a.credit_score <= 900:
        raise CalculationInputError("Credit score must be between 300 and 900.")

    rules = LENDING_RULES[a.loan_type]
    res = EligibilityResult(decision="likely_eligible", loan_type=a.loan_type,
                            requested_amount=a.requested_amount)
    res.documents = required_documents(a.loan_type, a.employment_type)
    blockers: list[tuple[str, str]] = []

    # Tenure limited by product maximum and by retirement age.
    if a.loan_type == "education":
        years_left_m = 10**6  # the student repays after studies; no retirement cap
    else:
        retire = RETIREMENT_AGE.get(a.employment_type, 60) + (10 if a.loan_type == "home" else 0)
        years_left_m = max((retire - a.age) * 12, 0)
    max_tenure = min(rules["max_tenure_m"], years_left_m)
    tenure = min(a.tenure_months, max_tenure) if max_tenure > 0 else 0
    if tenure <= 0:
        blockers.append(("Your age leaves no repayment period before retirement age for this loan.",
                         "रिटायरमेंट की उम्र से पहले इस लोन के लिए चुकाने का समय नहीं बचता।"))
    elif tenure < a.tenure_months:
        res.reasons.append(f"Tenure reduced to {tenure} months (product/age limit).")
        res.reasons_hi.append(f"अवधि घटाकर {tenure} महीने की गई (प्रोडक्ट/उम्र की सीमा)।")

    if a.monthly_income < rules["min_income"]:
        blockers.append((f"Monthly income is below the usual minimum of ₹{rules['min_income']:,} for this loan.",
                         f"मासिक आय इस लोन की सामान्य न्यूनतम ₹{rules['min_income']:,} से कम है।"))
    if a.credit_score is not None and a.credit_score < rules["min_score"]:
        blockers.append((f"Credit score {a.credit_score} is below the usual minimum of {rules['min_score']}.",
                         f"क्रेडिट स्कोर {a.credit_score} सामान्य न्यूनतम {rules['min_score']} से कम है।"))
    if a.credit_score is None:
        res.reasons.append("Credit score not provided; the highest indicative rate band is assumed.")
        res.reasons_hi.append("क्रेडिट स्कोर नहीं दिया गया; सबसे ऊँची संभावित ब्याज दर मानी गई है।")

    rate = _rate_for(rules, a.credit_score)
    res.indicative_rate_pct = rate
    cap_emi = max(a.monthly_income * rules["max_foir"] - a.existing_emi, 0.0)
    res.max_affordable_emi = round(cap_emi, 2)

    if rate is not None and tenure > 0:
        res.max_eligible_amount = max_principal_for_emi(cap_emi, rate, tenure)
        res.emi_for_requested = calculate_emi(a.requested_amount, rate, tenure)
        if a.monthly_income > 0:
            res.foir_after_loan_pct = round(100 * (a.existing_emi + res.emi_for_requested) / a.monthly_income, 1)
        if res.emi_for_requested > cap_emi:
            # Longest tenure needed to fit the request?
            for m in range(tenure, max_tenure + 1):
                if calculate_emi(a.requested_amount, rate, m) <= cap_emi:
                    res.suggested_tenure_months = m
                    break
            blockers_amount = (f"Requested EMI ₹{res.emi_for_requested:,.0f} is above what you can comfortably "
                               f"afford (₹{cap_emi:,.0f}/month, {int(rules['max_foir']*100)}% of income after existing EMIs).",
                               f"माँगी गई ईएमआई ₹{res.emi_for_requested:,.0f} आपकी सुविधाजनक सीमा (₹{cap_emi:,.0f}/माह) से ज़्यादा है।")
            res.reasons.append(blockers_amount[0])
            res.reasons_hi.append(blockers_amount[1])
            res.decision = "partially_eligible" if res.max_eligible_amount >= 0.3 * a.requested_amount else "unlikely"
            res.tips.append(f"You may qualify for about ₹{res.max_eligible_amount:,.0f} on this tenure.")
            res.tips_hi.append(f"इस अवधि पर आप लगभग ₹{res.max_eligible_amount:,.0f} के लिए पात्र हो सकते हैं।")
            if res.suggested_tenure_months:
                res.tips.append(f"A tenure of {res.suggested_tenure_months} months would fit your request (but costs more interest).")
                res.tips_hi.append(f"{res.suggested_tenure_months} महीने की अवधि में आपका अनुरोध फिट होगा (पर ब्याज ज़्यादा लगेगा)।")
    else:
        res.decision = "unlikely"

    if blockers:
        res.decision = "unlikely"
        for en, hi in blockers:
            res.reasons.append(en)
            res.reasons_hi.append(hi)
        if a.credit_score is not None and a.credit_score < rules["min_score"]:
            res.tips += ["Pay all dues on time for 6+ months and keep card usage under 30% of the limit to improve your score.",
                         "Consider a smaller amount, a secured loan, or adding a co-applicant."]
            res.tips_hi += ["6+ महीने सभी बकाया समय पर चुकाएँ और कार्ड का उपयोग सीमा के 30% से कम रखें।",
                            "छोटी राशि, सिक्योर्ड लोन या सह-आवेदक जोड़ने पर विचार करें।"]
    if res.decision == "likely_eligible":
        res.reasons.append("Your income and obligations support the requested EMI.")
        res.reasons_hi.append("आपकी आय और मौजूदा किस्तें माँगी गई ईएमआई के अनुकूल हैं।")
    if a.existing_emi > 0.3 * a.monthly_income > 0:
        res.tips.append("Closing a small existing loan first would raise your eligible amount.")
        res.tips_hi.append("पहले कोई छोटा मौजूदा लोन बंद करने से आपकी पात्र राशि बढ़ेगी।")
    return res


# ---------------------------------------------------------------------------
# Documents, glossary
# ---------------------------------------------------------------------------
_BASE_DOCS = [
    {"en": "PAN card", "hi": "पैन कार्ड"},
    {"en": "Aadhaar card (identity and address)", "hi": "आधार कार्ड (पहचान और पता)"},
    {"en": "Recent passport-size photograph", "hi": "हाल की पासपोर्ट साइज़ फोटो"},
    {"en": "Last 6 months bank statements", "hi": "पिछले 6 महीने का बैंक स्टेटमेंट"},
]
_EMPLOYMENT_DOCS = {
    "salaried": [
        {"en": "Last 3 months salary slips", "hi": "पिछले 3 महीने की सैलरी स्लिप"},
        {"en": "Form 16 or last 2 years ITR (if available)", "hi": "फॉर्म 16 या पिछले 2 साल का आईटीआर (यदि हो)"},
        {"en": "Employment ID / offer letter", "hi": "कंपनी आईडी / ऑफर लेटर"},
    ],
    "self_employed": [
        {"en": "Last 2-3 years ITR with computation of income", "hi": "पिछले 2-3 साल का आईटीआर"},
        {"en": "Business proof (GST registration / Udyam / shop licence)", "hi": "व्यवसाय का प्रमाण (जीएसटी / उद्यम / दुकान लाइसेंस)"},
        {"en": "Last 12 months business bank statements", "hi": "पिछले 12 महीने का बिज़नेस बैंक स्टेटमेंट"},
    ],
}
_LOAN_DOCS = {
    "home": [{"en": "Sale agreement and property papers", "hi": "बिक्री अनुबंध और संपत्ति के कागज़ात"},
             {"en": "Approved building plan and title documents", "hi": "मंज़ूर नक्शा और टाइटल दस्तावेज़"}],
    "vehicle": [{"en": "Vehicle quotation / proforma invoice", "hi": "वाहन का कोटेशन / प्रोफॉर्मा इनवॉइस"},
                {"en": "Valid driving licence", "hi": "वैध ड्राइविंग लाइसेंस"}],
    "business": [{"en": "Business plan or projections", "hi": "बिज़नेस प्लान या अनुमान"},
                 {"en": "Existing loan statements (if any)", "hi": "मौजूदा लोन स्टेटमेंट (यदि हों)"}],
    "education": [{"en": "Admission letter and fee structure", "hi": "एडमिशन लेटर और फीस विवरण"},
                  {"en": "Marksheets of last qualifying exam", "hi": "पिछली परीक्षा की मार्कशीट"},
                  {"en": "Co-applicant (parent/guardian) income proof", "hi": "सह-आवेदक (अभिभावक) का आय प्रमाण"}],
    "gold": [{"en": "Gold ornaments for valuation (no income proof usually needed)", "hi": "मूल्यांकन के लिए सोने के गहने"}],
    "personal": [],
}


def required_documents(loan_type: str, employment_type: str = "salaried") -> list[dict]:
    if loan_type not in LENDING_RULES:
        raise CalculationInputError(f"Unsupported loan type: {loan_type}")
    if loan_type == "gold":
        return _BASE_DOCS[:2] + _LOAN_DOCS["gold"]
    return _BASE_DOCS + _EMPLOYMENT_DOCS.get(employment_type, _EMPLOYMENT_DOCS["salaried"]) + _LOAN_DOCS[loan_type]


GLOSSARY: dict[str, dict[str, str]] = {
    "EMI": {"en": "Equated Monthly Instalment: the fixed amount you pay every month, part principal and part interest.",
            "hi": "हर महीने चुकाई जाने वाली तय किस्त, जिसमें मूलधन और ब्याज दोनों होते हैं।"},
    "Principal": {"en": "The amount you actually borrow.", "hi": "वह राशि जो आप वास्तव में उधार लेते हैं।"},
    "Tenure": {"en": "How long you take to repay the loan.", "hi": "लोन चुकाने की अवधि।"},
    "APR": {"en": "Annual Percentage Rate: the true yearly cost of the loan including fees, so offers can be compared fairly.",
            "hi": "लोन की असली सालाना लागत (शुल्क सहित), जिससे ऑफर की सही तुलना हो सके।"},
    "Processing fee": {"en": "One-time fee charged by the lender, usually 0.5%-3% of the loan, often deducted before disbursal.",
                       "hi": "लोन देने वाले का एकमुश्त शुल्क, आमतौर पर लोन का 0.5%-3%, अक्सर पहले ही काट लिया जाता है।"},
    "Foreclosure": {"en": "Closing the whole loan before the tenure ends. Check for foreclosure charges.",
                    "hi": "अवधि पूरी होने से पहले पूरा लोन बंद करना। फोरक्लोज़र शुल्क ज़रूर देखें।"},
    "Prepayment": {"en": "Paying part of the loan early to cut interest. Some lenders charge a fee.",
                   "hi": "ब्याज घटाने के लिए लोन का कुछ हिस्सा पहले चुकाना। कुछ लेंडर शुल्क लेते हैं।"},
    "Floating rate": {"en": "Interest rate that moves with a market benchmark, so your EMI or tenure can change.",
                      "hi": "बाज़ार के बेंचमार्क के साथ बदलने वाली ब्याज दर; इससे ईएमआई या अवधि बदल सकती है।"},
    "Fixed rate": {"en": "Interest rate that stays the same for the agreed period.", "hi": "तय अवधि तक न बदलने वाली ब्याज दर।"},
    "FOIR": {"en": "Fixed Obligations to Income Ratio: share of monthly income that goes to EMIs. Lenders prefer under 50%.",
             "hi": "मासिक आय का वह हिस्सा जो किस्तों में जाता है। लेंडर 50% से कम पसंद करते हैं।"},
    "Credit score": {"en": "A 300-900 number showing repayment history. 750+ usually gets the best rates.",
                     "hi": "चुकौती इतिहास दिखाने वाला 300-900 का नंबर। 750+ पर आमतौर पर सबसे अच्छी दर मिलती है।"},
    "Moratorium": {"en": "A period at the start when you may pay no EMI (interest usually still builds up).",
                   "hi": "शुरुआत की वह अवधि जब ईएमआई नहीं देनी पड़ती (ब्याज आमतौर पर फिर भी जुड़ता है)।"},
    "Collateral": {"en": "An asset (home, gold, vehicle) pledged to the lender as security.",
                   "hi": "लेंडर को सुरक्षा के रूप में गिरवी रखी गई संपत्ति (घर, सोना, वाहन)।"},
    "Penal charges": {"en": "Extra charges for late or missed EMIs.", "hi": "देर से या छूटी हुई ईएमआई पर अतिरिक्त शुल्क।"},
}


def explain_loan_text(text: str, language: str = "en", llm: Optional[LLMProvider] = None) -> dict:
    """Plain-language help for loan agreement text.

    Always returns the glossary terms found in the text. When a real LLM is configured
    it also returns a simple explanation (never advice, never new terms).
    """
    if not text or not text.strip():
        raise CalculationInputError("Paste some loan text to explain.")
    found = [{"term": term, **info} for term, info in GLOSSARY.items()
             if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE)]
    explanation = None
    if llm is not None and not isinstance(llm, MockLLMProvider):
        prompt = (
            "Explain this loan agreement text in simple English for a first-time borrower. "
            "Use only what the text says; do not add terms, advice or calculations. "
            "Point out any fees, penalties or conditions the borrower should notice, as bullet points. "
            f"If the text is unclear, say so.\n\nTEXT:\n{text[:6000]}"
        )
        try:
            explanation = llm.generate(prompt, temperature=0.1, max_tokens=700).strip()
            if language == "hi":
                explanation, _ = translate_text(explanation, "hi", llm)
        except LLMProviderError:
            explanation = None
    return {"explanation": explanation, "terms": found, "language": language,
            "note": None if explanation else "Connect an LLM (see .env) for a full plain-language explanation."}


# ---------------------------------------------------------------------------
# Offer comparison
# ---------------------------------------------------------------------------
class LoanOffer(BaseModel):
    name: str
    principal: float
    annual_rate_pct: float
    tenure_months: int
    processing_fee_pct: float = 0.0
    other_fees: float = 0.0


class OfferComparison(BaseModel):
    name: str
    emi: float
    total_interest: float
    fees: float
    total_cost: float  # interest + fees
    total_payment: float
    effective_apr_pct: float
    best: bool = False


def _effective_apr(net_disbursed: float, emi: float, months: int) -> float:
    """Annualised IRR of the loan after fees (bisection on the monthly rate)."""
    if net_disbursed <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        pv = emi * months if mid == 0 else emi * (1 - (1 + mid) ** -months) / mid
        lo, hi = (mid, hi) if pv > net_disbursed else (lo, mid)
    return round(((lo + hi) / 2) * 12 * 100, 2)


def compare_loan_offers(offers: list[LoanOffer]) -> list[OfferComparison]:
    if len(offers) < 2:
        raise CalculationInputError("Add at least two offers to compare.")
    results = []
    for offer in offers:
        emi = calculate_emi(offer.principal, offer.annual_rate_pct, offer.tenure_months)
        total_payment = round(emi * offer.tenure_months, 2)
        interest = round(total_payment - offer.principal, 2)
        fees = round(offer.principal * offer.processing_fee_pct / 100 + offer.other_fees, 2)
        results.append(OfferComparison(
            name=offer.name, emi=emi, total_interest=interest, fees=fees,
            total_cost=round(interest + fees, 2), total_payment=round(total_payment + fees, 2),
            effective_apr_pct=_effective_apr(offer.principal - fees, emi, offer.tenure_months)))
    cheapest = min(results, key=lambda r: r.total_cost)
    cheapest.best = True
    return results


# ---------------------------------------------------------------------------
# Prepayment planner (keep EMI, shorten tenure)
# ---------------------------------------------------------------------------
class PrepaymentPlan(BaseModel):
    emi: float
    original_months: int
    new_months: int
    months_saved: int
    original_interest: float
    new_interest: float
    interest_saved: float
    total_prepaid: float
    balance_original: list[float] = Field(default_factory=list)  # balance at the end of each year
    balance_new: list[float] = Field(default_factory=list)
    note: str = ("Assumes prepayments reduce the tenure and the EMI stays the same. "
                 "Some lenders charge a prepayment fee on fixed-rate loans; floating-rate home loans "
                 "for individuals usually have none.")


def prepayment_plan(principal: float, annual_rate_pct: float, tenure_months: int,
                    extra_monthly: float = 0.0, yearly_extra: float = 0.0,
                    lump_sum: float = 0.0, lump_sum_month: int = 12) -> PrepaymentPlan:
    """Interest and months saved by paying extra on top of the EMI."""
    if min(extra_monthly, yearly_extra, lump_sum) < 0:
        raise CalculationInputError("Extra payments cannot be negative.")
    emi = calculate_emi(principal, annual_rate_pct, tenure_months)
    r = annual_rate_pct / 12.0 / 100.0

    def simulate(with_extra: bool):
        balance, interest_total, prepaid, month, yearly = principal, 0.0, 0.0, 0, []
        while balance > 0.005 and month < tenure_months:
            month += 1
            interest = balance * r
            interest_total += interest
            balance = balance + interest - min(emi, balance + interest)
            if with_extra and balance > 0:
                extra = extra_monthly
                if month % 12 == 0:
                    extra += yearly_extra
                if month == lump_sum_month:
                    extra += lump_sum
                extra = min(extra, balance)
                balance -= extra
                prepaid += extra
            if month % 12 == 0 or balance <= 0.005:
                yearly.append(round(max(balance, 0.0), 2))
        return month, round(interest_total, 2), round(prepaid, 2), yearly

    m0, i0, _, y0 = simulate(False)
    m1, i1, prepaid, y1 = simulate(True)
    return PrepaymentPlan(emi=emi, original_months=m0, new_months=m1, months_saved=m0 - m1,
                          original_interest=i0, new_interest=i1, interest_saved=round(i0 - i1, 2),
                          total_prepaid=prepaid, balance_original=y0, balance_new=y1)


# ---------------------------------------------------------------------------
# Eligibility levers: what would change the answer?
# ---------------------------------------------------------------------------
class EligibilityLever(BaseModel):
    lever: str
    lever_hi: str
    max_eligible_amount: float
    change: float
    decision: str


def eligibility_levers(applicant: LoanApplicant) -> list[EligibilityLever]:
    """Re-run the (illustrative) eligibility rules with one change at a time."""
    base = check_loan_eligibility(applicant)
    rules = LENDING_RULES[applicant.loan_type]
    scenarios: list[tuple[str, str, LoanApplicant]] = []
    longer = min(applicant.tenure_months + 24, rules["max_tenure_m"])
    if longer > applicant.tenure_months:
        scenarios.append((f"Longer tenure ({longer} months)", f"लंबी अवधि ({longer} महीने)",
                          applicant.model_copy(update={"tenure_months": longer})))
    if applicant.existing_emi > 0:
        scenarios.append(("Close existing EMIs first", "पहले मौजूदा ईएमआई बंद करें",
                          applicant.model_copy(update={"existing_emi": 0.0})))
    if applicant.credit_score is None or applicant.credit_score < 750:
        scenarios.append(("Credit score 750+", "क्रेडिट स्कोर 750+",
                          applicant.model_copy(update={"credit_score": 760})))
    scenarios.append(("Add a co-applicant earning 50% of your income", "आपकी आय के 50% कमाने वाला सह-आवेदक जोड़ें",
                      applicant.model_copy(update={"monthly_income": applicant.monthly_income * 1.5})))
    out = [EligibilityLever(lever="Today", lever_hi="आज", max_eligible_amount=base.max_eligible_amount,
                            change=0.0, decision=base.decision)]
    for en, hi, changed in scenarios:
        res = check_loan_eligibility(changed)
        out.append(EligibilityLever(lever=en, lever_hi=hi, max_eligible_amount=res.max_eligible_amount,
                                    change=round(res.max_eligible_amount - base.max_eligible_amount, 2),
                                    decision=res.decision))
    return out


# ---------------------------------------------------------------------------
# Loan Pre-Flight: affordability + EMI stress test + rejection decoder
# (rules and thresholds come from knowledge/loan_preflight_rules.json - never from the LLM)
# ---------------------------------------------------------------------------
import json as _json
from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

from .utils import inr  # noqa: E402

PREFLIGHT_RULES_FILE = _Path(__file__).resolve().parent.parent / "knowledge" / "loan_preflight_rules.json"
SCORE_BANDS = ("750+", "700-749", "650-699", "<650", "unknown")


@_lru_cache(maxsize=1)
def preflight_rules() -> dict:
    return _json.loads(PREFLIGHT_RULES_FILE.read_text(encoding="utf-8"))


class LoanPreflightInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    monthly_income: float
    existing_emis: float = 0.0
    card_dues_monthly: float = 0.0  # what you pay towards credit cards each month
    monthly_expenses: Optional[float] = None  # rent, food, bills (not EMIs); optional
    loan_amount: float
    tenure_months: int
    annual_rate_pct: float
    employment_type: str = "salaried"
    loan_type: str = "personal"
    credit_score_range: str = "unknown"  # 750+ | 700-749 | 650-699 | <650 | unknown
    savings: Optional[float] = None
    card_limit: Optional[float] = None
    card_outstanding: Optional[float] = None
    has_overdue: bool = False
    recent_applications: int = 0


class CashFlowSnapshot(BaseModel):
    income: float
    expenses: float
    existing_emis: float
    card_dues: float
    new_emi: float
    buffer: float


class StressScenario(BaseModel):
    name: str
    detail: str
    emi: float
    dti_pct: float
    buffer: float
    status: str  # safe | tight | high_risk


class RiskFactor(BaseModel):
    code: str
    title: str
    detail: str
    actions: list[str]
    severity: str  # high | medium | low


class LoanPreflightResult(BaseModel):
    inputs: LoanPreflightInput
    emi: float
    total_interest: float
    total_payment: float
    dti_before_pct: float
    dti_after_pct: float
    before: CashFlowSnapshot
    after: CashFlowSnapshot
    buffer_after_pct: float
    status: str  # safe | tight | high_risk
    status_reason: str
    affordable_loan_amount: float
    scenarios: list[StressScenario] = Field(default_factory=list)
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    documents: list[dict] = Field(default_factory=list)
    explanations: dict = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    grounding: str = "general guidance from your inputs (no document used)"
    disclaimer: str = ""


STATUS_LABELS = {"safe": "Safe", "tight": "Tight", "high_risk": "High Risk"}


def _validate_preflight(inp: LoanPreflightInput) -> None:
    if inp.monthly_income <= 0:
        raise CalculationInputError("Monthly income must be more than zero.")
    if min(inp.existing_emis, inp.card_dues_monthly, inp.recent_applications) < 0 or (inp.monthly_expenses or 0) < 0 \
            or (inp.savings or 0) < 0:
        raise CalculationInputError("Amounts cannot be negative.")
    if inp.loan_amount <= 0 or inp.tenure_months <= 0:
        raise CalculationInputError("Loan amount and tenure must be more than zero.")
    if not 0 <= inp.annual_rate_pct <= 60:
        raise CalculationInputError("Enter the interest rate quoted to you (0-60% a year).")
    if inp.credit_score_range not in SCORE_BANDS:
        raise CalculationInputError("Credit score range must be one of: " + ", ".join(SCORE_BANDS))


def _status(dti_pct: float, buffer: float, income: float, expenses_known: bool) -> tuple[str, str]:
    levels = preflight_rules()["stress_levels"]
    buffer_pct = 100 * buffer / income
    if buffer < 0:
        return "high_risk", "Your monthly commitments would be more than your income."
    if dti_pct <= levels["safe"]["max_dti_pct"] and (not expenses_known or buffer_pct >= levels["safe"]["min_buffer_pct_of_income"]):
        return "safe", (f"EMIs would take {dti_pct:.0f}% of income (safe up to {levels['safe']['max_dti_pct']}%)"
                        + (f" and {buffer_pct:.0f}% of income would be left over." if expenses_known else "."))
    if dti_pct <= levels["tight"]["max_dti_pct"] and (not expenses_known or buffer_pct >= levels["tight"]["min_buffer_pct_of_income"]):
        return "tight", f"EMIs would take {dti_pct:.0f}% of income, leaving little room for surprises."
    return "high_risk", f"EMIs would take {dti_pct:.0f}% of income, above the {levels['tight']['max_dti_pct']}% level lenders usually accept."


def _snapshot(inp: LoanPreflightInput, new_emi: float, income: Optional[float] = None) -> CashFlowSnapshot:
    income = inp.monthly_income if income is None else income
    expenses = inp.monthly_expenses or 0.0
    buffer = income - expenses - inp.existing_emis - inp.card_dues_monthly - new_emi
    return CashFlowSnapshot(income=round(income, 2), expenses=round(expenses, 2), existing_emis=round(inp.existing_emis, 2),
                            card_dues=round(inp.card_dues_monthly, 2), new_emi=round(new_emi, 2), buffer=round(buffer, 2))


def _dti(inp: LoanPreflightInput, emi: float, income: Optional[float] = None) -> float:
    income = inp.monthly_income if income is None else income
    return round(100 * (inp.existing_emis + inp.card_dues_monthly + emi) / income, 1)


def loan_preflight(inp: LoanPreflightInput, shock_amount: Optional[float] = None) -> LoanPreflightResult:
    """Deterministic affordability check, stress test and risk factors for a proposed loan."""
    _validate_preflight(inp)
    rules = preflight_rules()
    summary = loan_summary(inp.loan_amount, inp.annual_rate_pct, inp.tenure_months)
    emi = summary.emi
    expenses_known = inp.monthly_expenses is not None
    before, after = _snapshot(inp, 0.0), _snapshot(inp, emi)
    dti_before, dti_after = _dti(inp, 0.0), _dti(inp, emi)
    status, reason = _status(dti_after, after.buffer, inp.monthly_income, expenses_known)

    safe_dti = rules["stress_levels"]["safe"]["max_dti_pct"]
    max_safe_emi = max(inp.monthly_income * safe_dti / 100 - inp.existing_emis - inp.card_dues_monthly, 0.0)
    affordable = max_principal_for_emi(max_safe_emi, inp.annual_rate_pct, inp.tenure_months)

    res = LoanPreflightResult(
        inputs=inp, emi=emi, total_interest=summary.total_interest, total_payment=summary.total_payment,
        dti_before_pct=dti_before, dti_after_pct=dti_after, before=before, after=after,
        buffer_after_pct=round(100 * after.buffer / inp.monthly_income, 1), status=status, status_reason=reason,
        affordable_loan_amount=round(affordable, -3) if affordable >= 1000 else round(affordable, 2),
        documents=required_documents(inp.loan_type if inp.loan_type in LENDING_RULES else "personal", inp.employment_type),
        explanations=dict(rules["explanations"]), disclaimer=rules["disclaimer"])
    if not expenses_known:
        res.assumptions.append("Monthly expenses were not given, so the leftover money may be overstated.")
    res.assumptions.append("Uses the interest rate you entered; the lender's actual rate may differ.")

    # Stress scenarios
    sc = rules["stress_scenarios"]
    res.scenarios.append(StressScenario(name="As planned", detail=f"{inp.annual_rate_pct}% for {inp.tenure_months} months",
                                        emi=emi, dti_pct=dti_after, buffer=after.buffer, status=status))
    hi_rate = inp.annual_rate_pct + sc["rate_increase_pct_points"]
    emi_hi = calculate_emi(inp.loan_amount, hi_rate, inp.tenure_months)
    snap = _snapshot(inp, emi_hi)
    res.scenarios.append(StressScenario(name=f"Rate +{sc['rate_increase_pct_points']}%", detail=f"if a floating rate rises to {hi_rate}%",
                                        emi=emi_hi, dti_pct=_dti(inp, emi_hi), buffer=snap.buffer,
                                        status=_status(_dti(inp, emi_hi), snap.buffer, inp.monthly_income, expenses_known)[0]))
    low_income = inp.monthly_income * (1 - sc["income_drop_pct"] / 100)
    snap = _snapshot(inp, emi, low_income)
    res.scenarios.append(StressScenario(name=f"Income -{sc['income_drop_pct']}%", detail=f"if income falls to {inr(low_income)}",
                                        emi=emi, dti_pct=_dti(inp, emi, low_income), buffer=snap.buffer,
                                        status=_status(_dti(inp, emi, low_income), snap.buffer, low_income, expenses_known)[0]))
    shock = shock_amount if shock_amount is not None else sc["default_shock_rupees"]
    if shock < 0:
        raise CalculationInputError("The emergency amount cannot be negative.")
    savings = inp.savings or 0.0
    gap = shock - savings - max(after.buffer, 0)
    if gap <= 0:
        detail, shock_status = f"{inr(shock)} emergency covered by savings and this month's leftover", status
    elif after.buffer > 0:
        months = gap / after.buffer
        detail, shock_status = (f"{inr(shock)} emergency leaves a {inr(gap)} gap; about {months:.1f} months of leftover "
                                "money to recover"), "tight" if months <= 3 else "high_risk"
    else:
        detail, shock_status = f"{inr(shock)} emergency leaves a {inr(gap)} gap with no monthly leftover to recover", "high_risk"
    res.scenarios.append(StressScenario(name="One-time emergency", detail=detail, emi=emi, dti_pct=dti_after,
                                        buffer=round(after.buffer - shock + savings, 2), status=shock_status))
    if inp.savings is None:
        res.assumptions.append("Savings were not given, so the emergency scenario assumes none.")

    # Risk factors from the user's own data only
    rf, reasons = rules["risk_factors"], {r["code"]: r for r in rules["rejection_reasons"]}

    def add(code: str, detail: str, severity: str) -> None:
        r = reasons[code]
        res.risk_factors.append(RiskFactor(code=code, title=r["title"], detail=detail, actions=r["actions"], severity=severity))

    if dti_after > rf["high_dti_pct"]:
        add("high_foir", f"With this loan, {dti_after:.0f}% of your income would go to EMIs and card dues.", "high")
    if inp.credit_score_range in ("<650", "650-699"):
        band = next(b for b in rules["credit_score_bands"] if b["band"] == inp.credit_score_range)
        add("low_credit_score", f"Your score range is {inp.credit_score_range}. {band['note']}", "high" if inp.credit_score_range == "<650" else "medium")
    if inp.card_limit and inp.card_outstanding is not None and inp.card_limit > 0:
        util = 100 * inp.card_outstanding / inp.card_limit
        if util > rf["high_card_utilization_pct"]:
            add("high_utilization", f"Your card usage is {util:.0f}% of the limit.", "medium")
    if inp.has_overdue:
        add("overdue_or_default", "You told us you have an overdue payment.", "high")
    if inp.recent_applications >= rf["many_recent_applications"]:
        add("too_many_enquiries", f"You have applied {inp.recent_applications} times recently.", "medium")
    if inp.employment_type == "self_employed":
        add("income_or_employment", "Self-employed applicants are usually asked for 2-3 years of ITR and business proof.", "low")

    # Actions
    if status != "safe" and affordable > 0:
        res.actions.append(f"A loan of about {inr(res.affordable_loan_amount)} at this rate and tenure would keep EMIs within "
                           f"{safe_dti}% of income.")
    longer = min(inp.tenure_months + 24, LENDING_RULES.get(inp.loan_type, LENDING_RULES["personal"])["max_tenure_m"])
    if status != "safe" and longer > inp.tenure_months:
        emi_long = calculate_emi(inp.loan_amount, inp.annual_rate_pct, longer)
        extra = emi_long * longer - emi * inp.tenure_months
        res.actions.append(f"A {longer}-month tenure lowers the EMI to {inr(emi_long)} but costs {inr(extra)} more interest.")
    for f in res.risk_factors:
        res.actions.append(f.actions[0])
    if not res.actions:
        res.actions.append("Your numbers look comfortable. Compare offers by total cost (interest + fees) before applying.")
    res.actions = list(dict.fromkeys(res.actions))
    return res


def loan_what_if(inp: LoanPreflightInput, tenure_months: Optional[int] = None, loan_amount: Optional[float] = None,
                 annual_rate_pct: Optional[float] = None, shock_amount: Optional[float] = None,
                 income_drop_pct: Optional[float] = None) -> dict:
    """Compare the plan with one or more changes (tenure, amount, rate, emergency, income drop)."""
    base = loan_preflight(inp)
    changes = {}
    if tenure_months is not None:
        changes["tenure_months"] = int(tenure_months)
    if loan_amount is not None:
        changes["loan_amount"] = float(loan_amount)
    if annual_rate_pct is not None:
        changes["annual_rate_pct"] = float(annual_rate_pct)
    if income_drop_pct is not None:
        if not 0 <= income_drop_pct < 100:
            raise CalculationInputError("Income drop must be between 0 and 99%.")
        changes["monthly_income"] = inp.monthly_income * (1 - income_drop_pct / 100)
    variant = loan_preflight(inp.model_copy(update=changes), shock_amount=shock_amount)

    def view(r: LoanPreflightResult) -> dict:
        return {"emi": r.emi, "total_interest": r.total_interest, "dti_after_pct": r.dti_after_pct,
                "monthly_leftover": r.after.buffer, "status": STATUS_LABELS[r.status],
                "emergency": next((s.model_dump() for s in r.scenarios if s.name == "One-time emergency"), None)}

    return {"current_plan": view(base), "what_if": view(variant), "changes": changes,
            "emi_difference": round(variant.emi - base.emi, 2),
            "interest_difference": round(variant.total_interest - base.total_interest, 2),
            "disclaimer": base.disclaimer}


class DecodedReason(BaseModel):
    code: str
    title: str
    explanation: str
    actions: list[str]
    quote: str
    location: str  # "rejection_letter.pdf page 1, line 9"


class RejectionDecode(BaseModel):
    source: str
    reasons: list[DecodedReason] = Field(default_factory=list)
    message: str = ""
    disclaimer: str = ""


def document_lines(filename: str, data: bytes) -> list[tuple[str, str]]:
    """(location, text) lines from a PDF, image (OCR) or text document."""
    name = _Path(filename or "document").name
    ext = _Path(name).suffix.lower()
    out: list[tuple[str, str]] = []
    if ext == ".pdf":
        import fitz  # PyMuPDF

        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = [p.get_text("text") for p in doc]
        if not any(p.strip() for p in pages):
            from .fintech import _read_document_text

            pages = [_read_document_text(".pdf", data) or ""]
    elif ext in (".png", ".jpg", ".jpeg"):
        from .fintech import _read_document_text

        pages = [_read_document_text(ext, data) or ""]
    elif ext == ".txt":
        pages = [data.decode("utf-8", errors="ignore")]
    else:
        raise CalculationInputError("Upload the letter or report as PDF, image or text.")
    for p, text in enumerate(pages, 1):
        for ln, line in enumerate(text.splitlines(), 1):
            if line.strip():
                out.append((f"{name} page {p}, line {ln}" if len(pages) > 1 or ext == ".pdf" else f"{name} line {ln}", line.strip()))
    return out


def decode_rejection(lines: list[tuple[str, str]], source: str = "your document") -> RejectionDecode:
    """Map phrases in a rejection letter / credit report to plain explanations and next steps (rules file)."""
    from .guardrails import redact_sensitive

    rules = preflight_rules()
    out = RejectionDecode(source=source, disclaimer=rules["disclaimer"])
    for reason in rules["rejection_reasons"]:
        pats = [re.compile(rf"(?<![a-z]){re.escape(p)}", re.IGNORECASE) for p in reason["patterns"]]
        hit = next(((loc, text) for loc, text in lines if any(p.search(text) for p in pats)), None)
        if hit:
            quote, _ = redact_sensitive(hit[1][:200])
            out.reasons.append(DecodedReason(code=reason["code"], title=reason["title"], explanation=reason["explanation"],
                                             actions=reason["actions"], quote=quote, location=hit[0]))
    if not out.reasons:
        out.message = ("No common rejection reason was recognised in this document. Ask the lender for the specific "
                       "reason in writing - you are entitled to know it.")
    return out
