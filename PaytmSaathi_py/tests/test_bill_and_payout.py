"""Hospital bill parsing and deterministic payout estimation."""

import pytest

from ai_engine.bill_extractor import categorize, parse_bill_text
from ai_engine.exceptions import CalculationInputError
from ai_engine.payout_calculator import estimate_payout, parse_amount
from ai_engine.schemas import (
    BillLineItem,
    ClaimContext,
    HospitalBill,
    MoneyValue,
    PolicyDetails,
    SubLimitEntry,
    WaitingPeriodEntry,
)

BILL_TEXT = """Apollo Care Hospital
Bill No: INV-1001
Patient Name: Ravi Kumar
Date of Admission: 10/03/2025
Date of Discharge: 13/03/2025
Diagnosis: Cataract
1. Room Rent (3 days x 6,000)   18,000
2. Doctor Consultation   4,500
3. Pharmacy - Medicines   7,250.50
4. Registration Charges   500
Grand Total   30,250.50
"""


def test_parse_bill_text_extracts_items_and_metadata():
    bill = parse_bill_text(BILL_TEXT, "bill.pdf")
    assert bill.hospital_name == "Apollo Care Hospital"
    assert bill.patient_name.startswith("Ravi Kumar")
    assert bill.days_of_stay == 3
    assert bill.stated_total == pytest.approx(30250.50)
    assert bill.computed_total == pytest.approx(30250.50)
    by_desc = {i.description.split("(")[0].strip(): i for i in bill.line_items}
    room = by_desc["Room Rent"]
    assert room.category == "room_rent" and room.quantity == 3 and room.unit_price == 6000
    assert by_desc["Registration Charges"].is_payable is False
    assert by_desc["Pharmacy - Medicines"].category == "medicines"


def test_categorize_does_not_match_short_keywords_inside_words():
    assert categorize("Particulars of services")[0] != "icu_rent"
    assert categorize("ICU charges")[0] == "icu_rent"
    assert categorize("Attendant food")[1] is False


def _details(**overrides):
    base = dict(
        sum_insured=MoneyValue(value=500000),
        room_rent_limit=MoneyValue(value=5000, source_clause="4.2", source_page=12),
        copay_percentage=10,
    )
    base.update(overrides)
    return PolicyDetails(**base)


def _bill(items, **kw):
    return HospitalBill(line_items=items, **kw)


def test_payout_full_walkthrough():
    bill = _bill(
        [
            BillLineItem(description="Room", category="room_rent", quantity=3, amount=18000),
            BillLineItem(description="Doctor", category="doctor_fees", amount=10000),
            BillLineItem(description="Medicines", category="medicines", amount=20000),
            BillLineItem(description="Registration", category="non_payable", amount=500, is_payable=False),
        ],
        days_of_stay=3,
    )
    est = estimate_payout(_details(), bill)
    assert est.bill_total == 48500
    assert est.non_payable_total == 500
    assert est.room_rent_deduction == pytest.approx(3000)
    assert est.proportionate_deduction == pytest.approx(1666.67, abs=0.01)
    assert est.copay_amount == pytest.approx(4333.33, abs=0.01)
    assert est.estimated_payout == pytest.approx(39000, abs=0.01)
    assert est.patient_pays == pytest.approx(9500, abs=0.01)
    assert est.eligibility_status == "likely_payable"
    assert any(s.clause_reference == "4.2" for s in est.steps)


def test_no_room_limit_means_no_cap_and_is_reported():
    bill = _bill([BillLineItem(description="Room", category="room_rent", amount=30000)], days_of_stay=3)
    est = estimate_payout(_details(room_rent_limit=None, copay_percentage=None), bill)
    assert est.estimated_payout == 30000
    assert "room_rent_limit" in est.missing_information


def test_sub_limit_caps_matching_treatment():
    details = _details(
        copay_percentage=None,
        sub_limits=[SubLimitEntry(condition="Cataract surgery", limit="₹40,000", source_clause="5.1")],
    )
    bill = _bill([BillLineItem(description="Surgery", category="surgery", amount=100000)],
                 procedure="Cataract surgery")
    est = estimate_payout(details, bill)
    assert est.estimated_payout == 40000
    assert est.sub_limit_deduction == 60000


def test_waiting_period_not_completed_blocks_payout():
    details = _details(
        copay_percentage=None,
        waiting_periods=[WaitingPeriodEntry(condition="Cataract", duration="2 years", duration_days=730)],
    )
    bill = _bill([BillLineItem(description="Surgery", category="surgery", amount=50000)],
                 procedure="Cataract surgery")
    est = estimate_payout(details, bill, ClaimContext(policy_age_days=100))
    assert est.eligibility_status == "likely_excluded"
    assert est.estimated_payout == 0
    assert est.payout_if_covered == 50000
    assert est.flags and est.flags_hi


def test_unknown_policy_age_needs_review_not_denial():
    details = _details(
        copay_percentage=None,
        waiting_periods=[WaitingPeriodEntry(condition="Cataract", duration_days=730)],
    )
    bill = _bill([BillLineItem(description="Surgery", category="surgery", amount=50000)], diagnosis="Cataract")
    est = estimate_payout(details, bill)
    assert est.eligibility_status == "needs_review"
    assert est.estimated_payout == 50000
    assert "policy_age_days" in est.missing_information


def test_sum_insured_remaining_caps_payout():
    bill = _bill([BillLineItem(description="Medicines", category="medicines", amount=50000)])
    est = estimate_payout(_details(copay_percentage=None), bill, ClaimContext(previously_claimed=480000))
    assert est.estimated_payout == 20000
    assert est.sum_insured_cap_deduction == 30000


def test_empty_bill_rejected():
    with pytest.raises(CalculationInputError):
        estimate_payout(_details(), HospitalBill())


def test_parse_amount_variants():
    assert parse_amount("₹50,000") == 50000
    assert parse_amount("Rs. 2 lakh") == 200000
    assert parse_amount("1% of sum insured", 500000) == 5000
    assert parse_amount("1% of sum insured") is None
    assert parse_amount("no amount here") is None
