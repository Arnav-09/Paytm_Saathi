"""Tools the Saathi assistant can call (exposed over MCP by ``mcp_server/server.py``).

Rules every tool follows:

* Deterministic and read-only. No tool changes data or sends anything anywhere.
* Inputs are validated by the underlying engine functions; a bad input returns
  ``{"error": ...}`` instead of raising, so the model can explain it.
* ``user_id`` and ``policy_id`` are *context parameters*. The app injects the real values and
  overwrites whatever the model sends, so the model can never read another user's data.
* Results are compact JSON, kept small so they fit the model's context.
"""

from __future__ import annotations

from datetime import date
from functools import wraps
from typing import Optional

from . import fintech, lending
from .exceptions import PaytmSaathiError

CONTEXT_PARAMS = ("user_id", "policy_id")  # set by the app, hidden from the model
MAX_RESULT_CHARS = 3500


def _safe(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except PaytmSaathiError as exc:
            return {"error": exc.message}
        except (ValueError, TypeError, KeyError) as exc:
            return {"error": f"Invalid input: {exc}"}
        except Exception as exc:  # never leak a traceback to the model
            return {"error": f"The tool failed: {type(exc).__name__}"}
        return result

    return wrapper


@_safe
def calculate_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> dict:
    """Monthly EMI, total interest and total payment for a loan.

    principal: loan amount in rupees. annual_rate_pct: yearly interest rate, e.g. 11.5.
    tenure_months: loan length in months.
    """
    return lending.loan_summary(principal, annual_rate_pct, tenure_months).model_dump()


@_safe
def prepayment_savings(principal: float, annual_rate_pct: float, tenure_months: int,
                       extra_monthly: float = 0.0, yearly_extra: float = 0.0, lump_sum: float = 0.0) -> dict:
    """Interest and months saved by paying extra on a loan (EMI stays the same).

    extra_monthly: extra rupees every month. yearly_extra: extra once a year (e.g. a bonus).
    lump_sum: one-time extra payment made after 12 months.
    """
    plan = lending.prepayment_plan(principal, annual_rate_pct, tenure_months, extra_monthly, yearly_extra, lump_sum)
    d = plan.model_dump(exclude={"balance_original", "balance_new", "note"})
    return d


@_safe
def loan_eligibility(loan_type: str, employment_type: str, age: int, monthly_income: float, existing_emi: float,
                     requested_amount: float, tenure_months: int, credit_score: int = 0) -> dict:
    """Indicative loan eligibility using illustrative rules (not a lender's decision).

    loan_type: personal, home, vehicle, business, education or gold.
    employment_type: salaried or self_employed. credit_score: 300-900, or 0 if unknown.
    """
    applicant = lending.LoanApplicant(loan_type=loan_type, employment_type=employment_type, age=age,
                                      monthly_income=monthly_income, existing_emi=existing_emi,
                                      credit_score=credit_score or None, requested_amount=requested_amount,
                                      tenure_months=tenure_months)
    res = lending.check_loan_eligibility(applicant)
    return res.model_dump(include={"decision", "indicative_rate_pct", "max_eligible_amount", "emi_for_requested",
                                   "foir_after_loan_pct", "reasons", "tips", "disclaimer"})


@_safe
def compare_loan_offers(offers: list[dict]) -> list[dict]:
    """Compare 2-4 loan offers by true cost (interest + fees).

    Each offer: {"name": str, "principal": number, "annual_rate_pct": number, "tenure_months": int,
    "processing_fee_pct": number (optional)}.
    """
    if not 2 <= len(offers) <= 4:
        raise ValueError("Give between 2 and 4 offers.")
    parsed = [lending.LoanOffer(**o) for o in offers]
    return [r.model_dump() for r in lending.compare_loan_offers(parsed)]


@_safe
def scan_message_for_scam(text: str, sender: str = "") -> dict:
    """Check an SMS/WhatsApp message for scam signs: wording, links, phone numbers and the sender.

    sender: optional SMS header (e.g. VM-HDFCBK) or phone number the message came from.
    """
    res = fintech.check_scam_message(text, sender or None)
    return {
        "risk": res.risk, "score": res.score, "reasons": res.reasons[:8], "claimed_companies": res.claimed_brands,
        "links": [{"domain": l.domain, "verdict": l.verdict, "why": l.reasons[:2]} for l in res.links],
        "phones": [{"number": p.number, "verdict": p.verdict} for p in res.phones],
        "sender": res.sender.model_dump(include={"verdict", "note"}) if res.sender else None,
        "advice": res.advice[:4],
    }


@_safe
def check_link(url: str) -> dict:
    """Check one web link for look-alike, shortened or suspicious patterns and against the blocklist.

    The result can never prove a link is safe; always open banking links from the official app.
    """
    from .linkcheck import analyse_link, detect_brands

    f = analyse_link(url, detect_brands(url))
    return {"domain": f.domain, "verdict": f.verdict, "reasons": f.reasons[:4], "score": f.score}


@_safe
def validate_payment_id(kind: str, value: str) -> dict:
    """Format-check an IFSC code or UPI ID. kind must be 'ifsc' or 'upi'. Never pass Aadhaar, PAN or card numbers."""
    if kind not in ("ifsc", "upi"):
        raise ValueError("kind must be 'ifsc' or 'upi'.")
    r = fintech.validate_field(kind, value)
    return {"valid": r.valid, "message": r.message}


GRIEVANCE = {
    "insurance": ["Insurer's Grievance Redressal Officer (first)", "IRDAI Bima Bharosa: bimabharosa.irdai.gov.in",
                  "Insurance Ombudsman (if no reply in 30 days or you are unhappy with the reply)", "Consumer court (last resort)"],
    "banking": ["Bank or app's customer support and nodal officer (first)", "RBI Complaint Management System: cms.rbi.org.in "
                "(if unresolved after 30 days)"],
    "fraud": ["Call 1930 (national cyber crime helpline) immediately", "Report at cybercrime.gov.in",
              "Tell your bank in writing and block cards/UPI", "Report numbers/messages on Sanchar Saathi 'Chakshu': sancharsaathi.gov.in"],
    "loan": ["Lender's Grievance Redressal Officer (first)", "RBI Complaint Management System: cms.rbi.org.in "
             "(if unresolved after 30 days)", "For harassment or threats also contact the police or cybercrime.gov.in"],
}


@_safe
def grievance_contacts(topic: str) -> dict:
    """Where to complain, in order. topic: insurance, banking, fraud or loan."""
    if topic not in GRIEVANCE:
        raise ValueError("topic must be one of: " + ", ".join(GRIEVANCE))
    return {"topic": topic, "steps": GRIEVANCE[topic]}


@_safe
def search_knowledge(query: str) -> dict:
    """Search Saathi's guides on insurance, claims, loans, payments, UPI safety and KYC. Returns the best passages."""
    from .kb import search_kb

    hits = search_kb(query, top_k=3)
    return {"passages": [{"source": h.section_name or h.source_filename, "text": h.text[:700]} for h in hits]}


@_safe
def search_my_policy(query: str, policy_id: str = "") -> dict:
    """Search the user's own uploaded health policy and return matching clauses with page numbers."""
    if not policy_id:
        return {"error": "The user has not selected a policy."}
    from .retriever import PolicyRetriever

    hits = PolicyRetriever().retrieve(policy_id, query, top_k=4)
    return {"clauses": [{"page": h.page_number, "clause": h.clause_reference, "text": h.text[:700]} for h in hits]}


@_safe
def my_policy_overview(policy_id: str = "") -> dict:
    """Summary of the user's policy: sum insured, limits, co-pay, exclusions, waiting periods and watch-outs."""
    if not policy_id:
        return {"error": "The user has not selected a policy."}
    from .policy_extractor import load_policy_details
    from .policy_insights import coverage_at_a_glance

    d = load_policy_details(policy_id)
    g = coverage_at_a_glance(d)
    return {"policy": d.policy_name, "insurer": d.insurer_name,
            "covered": [i.text for i in g.covered][:6], "not_covered": [i.text for i in g.not_covered][:8],
            "watch_outs": [i.text for i in g.watch_outs][:10], "not_found_in_policy": g.unknown}


@_safe
def policy_waiting_periods(policy_start_date: str, policy_id: str = "") -> dict:
    """When each waiting period in the user's policy ends. policy_start_date format: YYYY-MM-DD."""
    if not policy_id:
        return {"error": "The user has not selected a policy."}
    from .policy_extractor import load_policy_details
    from .policy_insights import waiting_period_timeline

    start = date.fromisoformat(policy_start_date)
    rows = waiting_period_timeline(load_policy_details(policy_id), start)
    return {"waiting_periods": [r.model_dump(include={"condition", "ends_on", "days_left", "completed"}) for r in rows]}


@_safe
def my_payment_tickets(user_id: str = "") -> dict:
    """The user's payment complaint tickets with status, evidence count and the next escalation step."""
    if not user_id:
        return {"error": "Not logged in."}
    out = []
    for t in fintech.list_dispute_tickets(user_id)[:10]:
        nxt = next((s for s in fintech.escalation_timeline(t) if not s.reached), None)
        out.append({"ticket_id": t.ticket_id, "issue": t.guidance.title, "status": t.status, "amount": t.amount,
                    "evidence_items": len(t.attachments), "next_step": f"{nxt.due_on}: {nxt.text}" if nxt else "All steps reached"})
    return {"tickets": out}


@_safe
def loan_preflight_check(monthly_income: float, loan_amount: float, tenure_months: int, annual_rate_pct: float,
                         existing_emis: float = 0.0, card_dues_monthly: float = 0.0, monthly_expenses: Optional[float] = None,
                         employment_type: str = "salaried", credit_score_range: str = "unknown",
                         savings: Optional[float] = None, shock_amount: Optional[float] = None,
                         recent_applications: int = 0, has_overdue: bool = False) -> dict:
    """Loan Pre-Flight from numbers the user gave in chat: EMI, total interest, debt-to-income, monthly leftover,
    Safe/Tight/High Risk stress test (rate rise, income drop, one-time emergency), risk factors, actions, documents.

    Use ONLY the user's numbers. Never assume an interest rate: if the user did not give one, ask.
    credit_score_range: 750+, 700-749, 650-699, <650 or unknown. shock_amount: one-time emergency (e.g. hospital bill).
    """
    from .session_tools import compact_preflight

    inp = lending.LoanPreflightInput(monthly_income=monthly_income, loan_amount=loan_amount, tenure_months=tenure_months,
                                     annual_rate_pct=annual_rate_pct, existing_emis=existing_emis,
                                     card_dues_monthly=card_dues_monthly, monthly_expenses=monthly_expenses,
                                     employment_type=employment_type, credit_score_range=credit_score_range,
                                     savings=savings, recent_applications=recent_applications, has_overdue=has_overdue)
    return compact_preflight(lending.loan_preflight(inp, shock_amount))


@_safe
def decode_rejection_text(text: str) -> dict:
    """Explain a loan rejection reason the user typed or pasted (e.g. 'high FOIR', 'multiple enquiries') and what to do next."""
    lines = [(f"your message line {i}", ln.strip()) for i, ln in enumerate((text or "").splitlines() or [text], 1) if ln.strip()]
    return lending.decode_rejection(lines, "your message").model_dump()


TOOLS = [calculate_emi, prepayment_savings, loan_eligibility, compare_loan_offers, scan_message_for_scam,
         check_link, validate_payment_id, grievance_contacts, search_knowledge, search_my_policy,
         my_policy_overview, policy_waiting_periods, my_payment_tickets, loan_preflight_check, decode_rejection_text]
TOOL_FUNCS = {fn.__name__: fn for fn in TOOLS}
