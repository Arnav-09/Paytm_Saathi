"""Lending maths/eligibility and fintech validators, disputes and scam check."""

import pytest

from ai_engine import fintech, lending
from ai_engine.exceptions import CalculationInputError, PaytmSaathiError


# ----------------------------- lending ------------------------------------
def test_emi_known_value_and_zero_rate():
    assert lending.calculate_emi(1_000_000, 12, 12) == pytest.approx(88848.79, abs=0.01)
    assert lending.calculate_emi(120_000, 0, 12) == 10000


def test_schedule_closes_to_zero_and_repays_principal():
    rows = lending.amortization_schedule(500_000, 11, 36)
    assert rows[-1].balance == 0
    assert sum(r.principal for r in rows) == pytest.approx(500_000, abs=0.5)


def test_invalid_emi_inputs():
    with pytest.raises(CalculationInputError):
        lending.calculate_emi(0, 10, 12)
    with pytest.raises(CalculationInputError):
        lending.calculate_emi(1000, 10, 0)


def _applicant(**kw):
    base = dict(loan_type="personal", employment_type="salaried", age=30, monthly_income=100000,
                existing_emi=0, credit_score=780, requested_amount=500000, tenure_months=36)
    base.update(kw)
    return lending.LoanApplicant(**base)


def test_strong_applicant_is_likely_eligible():
    res = lending.check_loan_eligibility(_applicant())
    assert res.decision == "likely_eligible"
    assert res.indicative_rate_pct == 11.5
    assert res.documents


def test_low_credit_score_is_unlikely_with_reason():
    res = lending.check_loan_eligibility(_applicant(credit_score=600))
    assert res.decision == "unlikely"
    assert any("Credit score" in r for r in res.reasons)
    assert len(res.reasons) == len(res.reasons_hi)


def test_request_above_capacity_is_partial_with_max_amount():
    res = lending.check_loan_eligibility(
        _applicant(monthly_income=30000, credit_score=750, requested_amount=1_000_000))
    assert res.decision == "partially_eligible"
    assert 0 < res.max_eligible_amount < 1_000_000


def test_bad_inputs_rejected():
    with pytest.raises(CalculationInputError):
        lending.check_loan_eligibility(_applicant(loan_type="crypto"))
    with pytest.raises(CalculationInputError):
        lending.check_loan_eligibility(_applicant(credit_score=1000))


def test_compare_offers_marks_one_best_and_includes_fees():
    offers = [
        lending.LoanOffer(name="A", principal=500000, annual_rate_pct=11, tenure_months=36, processing_fee_pct=1),
        lending.LoanOffer(name="B", principal=500000, annual_rate_pct=10.5, tenure_months=36, processing_fee_pct=3),
    ]
    results = lending.compare_loan_offers(offers)
    assert sum(r.best for r in results) == 1
    assert results[0].effective_apr_pct >= 11
    with pytest.raises(CalculationInputError):
        lending.compare_loan_offers(offers[:1])


def test_documents_and_explain_without_llm():
    docs = lending.required_documents("home", "self_employed")
    assert all({"en", "hi"} <= set(d) for d in docs)
    out = lending.explain_loan_text("A processing fee applies and foreclosure charges may apply.", "en", None)
    assert {t["term"] for t in out["terms"]} >= {"Processing fee", "Foreclosure"}
    assert out["explanation"] is None


# ----------------------------- fintech ------------------------------------
def _valid_aadhaar():
    base = "23456789012"
    return next(base + str(d) for d in range(10) if fintech._verhoeff_valid(base + str(d)))


def test_identifier_validation():
    assert fintech.validate_field("pan", "ABCDE1234F").valid
    assert not fintech.validate_field("pan", "abcde1234").valid
    assert fintech.validate_field("ifsc", "SBIN0001234").valid
    assert fintech.validate_field("upi", "ravi.k@okaxis").valid
    assert not fintech.validate_field("upi", "ravi@@x").valid
    assert fintech.validate_field("mobile", "+91 98765 43210").valid
    assert not fintech.validate_field("mobile", "1234567890").valid
    assert fintech.validate_field("aadhaar", _valid_aadhaar()).valid
    good = _valid_aadhaar()
    wrong = good[:-1] + str((int(good[-1]) + 1) % 10)
    assert not fintech.validate_field("aadhaar", wrong).valid
    with pytest.raises(CalculationInputError):
        fintech.validate_field("passport", "x")


def test_aadhaar_is_masked_in_output():
    number = _valid_aadhaar()
    result = fintech.validate_field("aadhaar", number)
    assert number not in result.model_dump_json()
    assert result.masked_value.endswith(number[-4:]) and result.masked_value.startswith("XXXXXXXX")


def test_kyc_assessment_ready_and_missing():
    ok = fintech.assess_kyc("full", "9876543210", _valid_aadhaar(), "ABCDE1234F", True)
    assert ok.ready and not ok.missing
    bad = fintech.assess_kyc("full", "9876543210", "", "BAD", True)
    assert not bad.ready and len(bad.missing) == len(bad.missing_hi) == 2
    with pytest.raises(CalculationInputError):
        fintech.assess_kyc("platinum")


def test_dispute_classification_english_and_hindi():
    assert fintech.classify_dispute("Money debited but payment failed")[0] == "money_debited_failed"
    assert fintech.classify_dispute("Someone did fraud, I shared OTP, not me")[0] == "unauthorized_fraud"
    assert fintech.classify_dispute("पैसा कट गया और पेमेंट फेल हो गया")[0] == "money_debited_failed"
    assert fintech.classify_dispute("hello there") == ("other", "low")
    with pytest.raises(CalculationInputError):
        fintech.classify_dispute("   ")


def test_ticket_roundtrip_and_fraud_is_urgent(tmp_path):
    ticket = fintech.create_dispute_ticket("This is a fraud, hacked account", 5000, "UTR123", directory=str(tmp_path))
    assert ticket.guidance.urgent
    assert any("1930" in step for step in ticket.guidance.steps)
    loaded = fintech.get_dispute_ticket(ticket.ticket_id, directory=str(tmp_path))
    assert loaded.ticket_id == ticket.ticket_id and loaded.amount == 5000
    with pytest.raises(CalculationInputError):
        fintech.get_dispute_ticket("../../etc/passwd", directory=str(tmp_path))
    with pytest.raises(PaytmSaathiError):
        fintech.get_dispute_ticket("DSP-20250101-ABCDEF", directory=str(tmp_path))


def test_scam_check_levels():
    scam = fintech.check_scam_message("Your KYC will expire today! Share OTP and click bit.ly/abc to avoid block")
    assert scam.risk == "high" and len(scam.reasons) == len(scam.reasons_hi)
    genuine = fintech.check_scam_message("123456 is your OTP. Do not share it with anyone.")
    assert genuine.risk == "low"
    assert fintech.check_scam_message("Lunch at 1?").risk == "low"
    with pytest.raises(CalculationInputError):
        fintech.check_scam_message("")
