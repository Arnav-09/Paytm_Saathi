"""Feature upgrades: policy insights, prepayment, eligibility levers, complaints, scam spans, LLM presets."""

from datetime import date, datetime, timezone

import pytest

from ai_engine import fintech, lending
from ai_engine.exceptions import CalculationInputError
from ai_engine.llm_provider import MockLLMProvider, OpenAICompatibleProvider, get_llm_provider, provider_status
from ai_engine.policy_insights import (
    claim_documents,
    coverage_at_a_glance,
    room_rent_what_if,
    waiting_period_timeline,
)
from ai_engine.schemas import (
    BillLineItem,
    ExclusionEntry,
    HospitalBill,
    MoneyValue,
    PolicyDetails,
    WaitingPeriodEntry,
)


def _details(**kw):
    base = dict(
        sum_insured=MoneyValue(value=500000, source_page=2),
        room_rent_limit=MoneyValue(value=5000, source_clause="4.2", source_page=12),
        copay_percentage=10,
        exclusions=[ExclusionEntry(condition="Cosmetic surgery", source_clause="7.1", source_page=20)],
        waiting_periods=[WaitingPeriodEntry(condition="Cataract", duration_days=730),
                         WaitingPeriodEntry(condition="Initial", duration_days=30)],
    )
    base.update(kw)
    return PolicyDetails(**base)


# ----------------------------- insurance ----------------------------------
def test_coverage_at_a_glance_groups_and_cites():
    g = coverage_at_a_glance(_details())
    assert any("500,000" in i.text for i in g.covered)
    assert g.not_covered[0].clause == "7.1" and g.not_covered[0].page == 20
    texts = " ".join(i.text for i in g.watch_outs)
    assert "5,000/day" in texts and "10%" in texts and "Cataract" in texts
    assert all(i.text_hi for i in g.covered + g.not_covered + g.watch_outs)


def test_waiting_period_timeline():
    rows = waiting_period_timeline(_details(), date(2025, 1, 1), today=date(2025, 3, 2))
    by = {r.condition: r for r in rows}
    assert by["Initial"].completed and by["Initial"].days_left == 0
    assert not by["Cataract"].completed and by["Cataract"].ends_on == "2027-01-01"
    assert 0 < by["Cataract"].progress < 1
    with pytest.raises(CalculationInputError):
        waiting_period_timeline(_details(), date(2999, 1, 1))


def test_room_what_if_shows_saving_within_limit():
    bill = HospitalBill(days_of_stay=3, line_items=[
        BillLineItem(description="Room", category="room_rent", quantity=3, amount=24000),  # 8000/day
        BillLineItem(description="Surgeon", category="surgery", amount=40000),
    ])
    options = room_rent_what_if(_details(copay_percentage=None), bill)
    mine = next(o for o in options if o.label == "Your room")
    within = next(o for o in options if o.room_rent_per_day == 5000)
    assert mine.room_rent_per_day == 8000 and mine.within_limit is False
    assert within.within_limit and within.you_pay < mine.you_pay
    assert within.you_pay == pytest.approx(0)  # fully covered within the limit, no co-pay


def test_room_what_if_needs_days():
    with pytest.raises(CalculationInputError):
        room_rent_what_if(_details(), HospitalBill(line_items=[BillLineItem(description="x", amount=1)]))


def test_claim_documents_tailored():
    bill = HospitalBill(diagnosis="Fracture after road accident",
                        line_items=[BillLineItem(description="OT", category="surgery", amount=1)])
    cashless = [d["en"] for d in claim_documents("cashless", bill)]
    reimb = [d["en"] for d in claim_documents("reimbursement", bill)]
    assert any("Pre-authorisation" in d for d in cashless)
    assert any("Cancelled cheque" in d for d in reimb)
    assert any("FIR" in d for d in reimb) and any("Operation theatre" in d for d in reimb)
    with pytest.raises(CalculationInputError):
        claim_documents("other")


# ----------------------------- lending ------------------------------------
def test_prepayment_saves_interest_and_months():
    plan = lending.prepayment_plan(1_000_000, 9, 240, yearly_extra=50_000)
    assert plan.original_months == 240
    assert plan.new_months < 240 and plan.months_saved == 240 - plan.new_months
    assert plan.interest_saved > 0 and plan.new_interest < plan.original_interest
    assert plan.balance_new[-1] == 0
    none = lending.prepayment_plan(1_000_000, 9, 240)
    assert none.months_saved == 0 and none.interest_saved == pytest.approx(0, abs=1)
    with pytest.raises(CalculationInputError):
        lending.prepayment_plan(1_000_000, 9, 240, extra_monthly=-1)


def test_eligibility_levers_show_improvements():
    applicant = lending.LoanApplicant(loan_type="personal", monthly_income=40000, existing_emi=8000,
                                      credit_score=700, requested_amount=800000, tenure_months=36)
    levers = {lv.lever: lv for lv in lending.eligibility_levers(applicant)}
    assert levers["Today"].change == 0
    assert levers["Close existing EMIs first"].change > 0
    assert levers["Credit score 750+"].change > 0
    assert any(k.startswith("Longer tenure") and v.change > 0 for k, v in levers.items())


# ----------------------------- fintech ------------------------------------
def test_tickets_listed_per_user_with_dates(tmp_path):
    d = str(tmp_path)
    mine = fintech.create_dispute_ticket("money debited but payment failed", 2000, "UTR123456",
                                         directory=d, user_id="U1", transaction_date="2026-11-01")
    fintech.create_dispute_ticket("refund not received", directory=d, user_id="U2")
    listed = fintech.list_dispute_tickets("U1", directory=d)
    assert [t.ticket_id for t in listed] == [mine.ticket_id]
    assert listed[0].transaction_date == "2026-11-01"


def test_escalation_timeline_dates():
    ticket = fintech.DisputeTicket(ticket_id="DSP-20261101-ABCDEF", dispute_type="money_debited_failed",
                                   description="x", created_at="2026-11-01T10:00:00+00:00",
                                   guidance=fintech.dispute_guidance("money_debited_failed"))
    steps = fintech.escalation_timeline(ticket, today=datetime(2026, 11, 3, tzinfo=timezone.utc))
    assert [s.day for s in steps] == [0, 1, 5, 30]
    assert steps[2].due_on == "2026-11-06" and not steps[2].reached and steps[1].reached


def test_complaint_draft_prefilled_and_safe():
    ticket = fintech.DisputeTicket(ticket_id="DSP-20261101-ABCDEF", dispute_type="unauthorized_fraud",
                                   description="Money taken without my consent", amount=5000,
                                   transaction_id="UTR999", transaction_date="2026-11-01",
                                   created_at="2026-11-01T10:00:00+00:00",
                                   guidance=fintech.dispute_guidance("unauthorized_fraud"))
    text = fintech.draft_complaint(ticket, "Ravi Kumar", "ombudsman", "XXXXXX3210")
    assert "Rs. 5,000.00" in text and "UTR999" in text and "2026-11-01" in text
    assert "Integrated Ombudsman" in text and "1930" in text and "Ravi Kumar" in text
    assert "XXXXXX3210" in text
    with pytest.raises(CalculationInputError):
        fintech.draft_complaint(ticket, "x", "police")


def test_scam_spans_point_at_risky_words():
    msg = "Your KYC will expire today. Share OTP now: bit.ly/x"
    res = fintech.check_scam_message(msg)
    found = [msg[a:b].lower() for a, b in res.spans]
    assert any("otp" in f for f in found) and any("bit.ly/" in f for f in found)
    assert all(a < b for a, b in res.spans)
    assert all(res.spans[i][1] <= res.spans[i + 1][0] for i in range(len(res.spans) - 1))  # merged, ordered


# ----------------------------- LLM presets --------------------------------
def test_groq_preset_without_key_falls_back_to_demo(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert isinstance(get_llm_provider(), MockLLMProvider)
    status = provider_status()
    assert not status["connected"] and "GROQ_API_KEY" in status["problem"]


def test_groq_preset_with_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    llm = get_llm_provider()
    assert isinstance(llm, OpenAICompatibleProvider)
    assert llm.base_url == "https://api.groq.com/openai/v1" and llm.label == "Groq"
    assert provider_status()["connected"]


# ----------------------------- regressions --------------------------------
@pytest.mark.parametrize("text,days", [
    ("30 days", 30), ("24 months", 730), ("36 months", 1095), ("9 months", 274),
    ("2 years", 730), ("3 yrs", 1095), ("48", 48), ("no number", None),
])
def test_duration_to_days(text, days):
    from ai_engine.schemas import duration_to_days

    assert duration_to_days(text) == days


def test_waiting_period_days_filled_from_months():
    w = WaitingPeriodEntry(condition="Pre-existing diseases", duration="36 months", duration_days=None)
    assert w.duration_days == 1095
    assert WaitingPeriodEntry(condition="x", duration_days="24 months").duration_days == 730


def test_registry_collection_name_is_valid_for_chromadb():
    import re

    from ai_engine.vector_store import POLICY_REGISTRY_COLLECTION

    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]", POLICY_REGISTRY_COLLECTION)
