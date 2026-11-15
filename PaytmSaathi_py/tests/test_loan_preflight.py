"""Loan Pre-Flight: EMI formula, affordability, Safe/Tight/High Risk stress test, scenarios, risk factors,
what-if, configurable rules, rejection decoder."""

import json
from pathlib import Path

import pytest

from ai_engine import lending
from ai_engine.exceptions import CalculationInputError

SAMPLES = Path(__file__).resolve().parent.parent / "samples" / "statements"


def plan(**kw):
    base = dict(monthly_income=85000, existing_emis=12500, card_dues_monthly=5000, monthly_expenses=40000,
                loan_amount=500000, tenure_months=36, annual_rate_pct=13.5, credit_score_range="700-749", savings=30000)
    base.update(kw)
    return lending.LoanPreflightInput(**base)


# ----------------------------- EMI ------------------------------------------
@pytest.mark.parametrize("p,rate,n", [(500000, 13.5, 36), (1000000, 12, 12), (2500000, 8.75, 240), (300000, 0.5, 6)])
def test_emi_matches_the_formula(p, rate, n):
    r = rate / 12 / 100
    expected = p * r * (1 + r) ** n / ((1 + r) ** n - 1)  # EMI = P x r x (1+r)^n / ((1+r)^n - 1)
    assert lending.calculate_emi(p, rate, n) == pytest.approx(expected, abs=0.01)


def test_zero_rate_and_totals():
    assert lending.calculate_emi(120000, 0, 12) == 10000
    s = lending.loan_summary(500000, 13.5, 36)
    assert s.total_payment == pytest.approx(s.emi * 36, abs=1) and s.total_interest == pytest.approx(s.total_payment - 500000)


# ----------------------------- affordability --------------------------------
def test_demo_plan_is_tight_with_cash_flow_before_after():
    r = lending.loan_preflight(plan())
    assert r.emi == pytest.approx(16967.64, abs=0.01)
    assert r.dti_before_pct == pytest.approx(20.6) and r.dti_after_pct == pytest.approx(40.6)
    assert r.before.buffer == 27500 and r.after.buffer == pytest.approx(10532.36)
    assert r.status == "tight" and lending.STATUS_LABELS[r.status] == "Tight"


@pytest.mark.parametrize("changes,status", [
    (dict(loan_amount=200000), "safe"),
    (dict(loan_amount=500000), "tight"),
    (dict(loan_amount=1500000), "high_risk"),
    (dict(monthly_expenses=70000), "high_risk"),  # commitments exceed income
])
def test_stress_status_levels(changes, status):
    assert lending.loan_preflight(plan(**changes)).status == status


def test_affordable_amount_keeps_dti_at_safe_limit():
    r = lending.loan_preflight(plan())
    safe_dti = lending.preflight_rules()["stress_levels"]["safe"]["max_dti_pct"]
    emi = lending.calculate_emi(r.affordable_loan_amount, 13.5, 36)
    assert 100 * (12500 + 5000 + emi) / 85000 <= safe_dti + 0.1


def test_missing_expenses_is_flagged_as_an_assumption():
    r = lending.loan_preflight(plan(monthly_expenses=None))
    assert any("expenses were not given" in a for a in r.assumptions)


def test_scenarios():
    r = lending.loan_preflight(plan())
    s = {x.name.split()[0]: x for x in r.scenarios}
    assert s["Rate"].emi > r.emi and s["Income"].status == "high_risk"
    assert s["One-time"].status == "high_risk" and "5.6 months" in s["One-time"].detail
    rich = lending.loan_preflight(plan(savings=500000))
    assert next(x for x in rich.scenarios if x.name == "One-time emergency").detail.endswith("covered by savings and this month's leftover")


def test_risk_factors_come_only_from_user_data():
    r = lending.loan_preflight(plan(credit_score_range="<650", card_limit=100000, card_outstanding=60000, has_overdue=True,
                                    recent_applications=4, employment_type="self_employed", loan_amount=1200000))
    codes = {f.code for f in r.risk_factors}
    assert codes == {"high_foir", "low_credit_score", "high_utilization", "overdue_or_default", "too_many_enquiries", "income_or_employment"}
    assert lending.loan_preflight(plan(credit_score_range="750+", loan_amount=200000)).risk_factors == []


def test_what_if_tenure_and_emergency():
    w = lending.loan_what_if(plan(), tenure_months=60)
    assert w["what_if"]["emi"] < w["current_plan"]["emi"] and w["interest_difference"] > 0
    h = lending.loan_what_if(plan(), shock_amount=150000)
    assert h["what_if"]["emergency"]["status"] == "high_risk" and "1,50,000" in h["what_if"]["emergency"]["detail"]
    with pytest.raises(CalculationInputError):
        lending.loan_what_if(plan(), income_drop_pct=120)


@pytest.mark.parametrize("bad", [dict(monthly_income=0), dict(loan_amount=-1), dict(annual_rate_pct=80),
                                 dict(credit_score_range="excellent"), dict(existing_emis=-5)])
def test_validation(bad):
    with pytest.raises(CalculationInputError):
        lending.loan_preflight(plan(**bad))


def test_rules_are_configurable(monkeypatch):
    rules = json.loads(json.dumps(lending.preflight_rules()))
    rules["stress_levels"]["tight"]["max_dti_pct"] = 35  # stricter lender view
    monkeypatch.setattr(lending, "preflight_rules", lambda: rules)
    assert lending.loan_preflight(plan()).status == "high_risk"


def test_disclaimer_is_always_present():
    r = lending.loan_preflight(plan())
    assert r.disclaimer.startswith("This is not a loan approval, credit decision, or guarantee of eligibility.")
    assert "general guidance" in r.grounding


# ----------------------------- rejection decoder ----------------------------
def test_decode_sample_rejection_letter_with_exact_lines():
    if not (SAMPLES / "loan_rejection_letter.pdf").exists():
        import runpy

        runpy.run_path(str(SAMPLES.parent / "make_statement_samples.py"), run_name="__main__")
    lines = lending.document_lines("loan_rejection_letter.pdf", (SAMPLES / "loan_rejection_letter.pdf").read_bytes())
    d = lending.decode_rejection(lines, "loan_rejection_letter.pdf")
    by = {r.code: r for r in d.reasons}
    assert set(by) == {"high_foir", "too_many_enquiries", "low_credit_score"}
    assert by["high_foir"].location.startswith("loan_rejection_letter.pdf page 1, line") and "FOIR" in by["high_foir"].quote
    assert all(r.actions for r in d.reasons)


def test_decode_text_and_unknown():
    d = lending.decode_rejection([("line 1", "Account shows 90 DPD and was written off")])
    assert [r.code for r in d.reasons] == ["overdue_or_default"]
    none = lending.decode_rejection([("line 1", "Thank you for your interest.")])
    assert none.reasons == [] and "specific reason" in none.message


def test_document_lines_rejects_unsupported_type():
    with pytest.raises(CalculationInputError):
        lending.document_lines("x.docx", b"PK")
