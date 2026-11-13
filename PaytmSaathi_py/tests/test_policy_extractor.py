"""Tests for policy_extractor.py (Feature 2)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai_engine.document_processor import PageContent
from ai_engine.exceptions import ValidationError as PSAValidationError
from ai_engine.policy_extractor import PolicyExtractor, load_policy_details
from ai_engine.schemas import PolicyDetails


class ExtractionLLM:
    """Mock LLM returning a valid, partially-filled extraction JSON."""

    def generate(self, *args, **kwargs):
        return "not used"

    def generate_structured(self, prompt, schema, **kwargs):
        return json.loads(
            '{"insurer_name": "Test Insurer", "policy_name": "Test Health Policy", '
            '"sum_insured": {"value": "₹5,00,000", "source_page": 3, "source_clause": "Section 2", '
            '"quoted_text": "Sum insured is INR 5,00,000", "is_explicit": true}, '
            '"room_rent_limit": null, "copay_percentage": "10%", '
            '"exclusions": [{"condition": "War", "source_page": 9, "source_clause": "Exclusion 4", '
            '"quoted_text": "War is excluded"}], '
            '"unknown_fields": ["icu_room_rent_limit", "pre_hospitalization_coverage_period"]}'
        )


class BadJSONLLM:
    """Mock LLM returning data that fails Pydantic validation - must be handled safely."""

    def generate_structured(self, prompt, schema, **kwargs):
        # sum_insured should be a MoneyValue object or null, not a string.
        # This will fail Pydantic validation -> extraction_validation_failed path.
        return {"sum_insured": "not an object", "insurer_name": "X"}


class EmptyDictLLM:
    """Mock LLM returning an empty object - all fields should be null."""

    def generate_structured(self, prompt, schema, **kwargs):
        return {}


def _pages():
    return [
        PageContent(page_number=3, page_text="Sum insured is INR 5,00,000 under Section 2.", source_filename="p.pdf"),
        PageContent(page_number=9, page_text="Exclusion 4: War is excluded.", source_filename="p.pdf"),
    ]


def test_extraction_validates_and_coerces_values():
    extractor = PolicyExtractor(llm_provider=ExtractionLLM())
    details = extractor.extract(_pages(), source_filename="p.pdf")

    assert isinstance(details, PolicyDetails)
    assert details.insurer_name == "Test Insurer"
    assert details.policy_name == "Test Health Policy"
    assert details.sum_insured is not None
    assert details.sum_insured.value == 500000.0  # "₹5,00,000" coerced
    assert details.sum_insured.currency == "INR"
    assert details.sum_insured.source_page == 3
    assert details.sum_insured.is_explicit is True
    assert details.copay_percentage == 10.0  # "10%" coerced
    assert len(details.exclusions) == 1
    assert details.exclusions[0].condition == "War"
    assert details.exclusions[0].source_page == 9
    assert "icu_room_rent_limit" in details.unknown_fields
    assert details.source_filename == "p.pdf"


def test_invalid_llm_json_is_handled_safely():
    extractor = PolicyExtractor(llm_provider=BadJSONLLM())
    details = extractor.extract(_pages(), source_filename="p.pdf")

    assert isinstance(details, PolicyDetails)
    # Safe fallback: nothing assumed, fields null, unknown_fields populated.
    assert details.insurer_name is None
    assert details.sum_insured is None
    assert details.copay_percentage is None
    assert "extraction_validation_failed" in details.unknown_fields


def test_empty_extraction_all_fields_null():
    extractor = PolicyExtractor(llm_provider=EmptyDictLLM())
    details = extractor.extract(_pages(), source_filename="p.pdf")

    assert isinstance(details, PolicyDetails)
    assert details.insurer_name is None
    assert details.sum_insured is None
    assert details.exclusions == []
    assert set(details.unknown_fields) == {
        "insurer_name",
        "policy_name",
        "sum_insured",
        "room_rent_limit",
        "copay_percentage",
        "exclusions",
        "waiting_periods",
        "claim_conditions",
    }


@patch("ai_engine.policy_extractor.load_policy_details")
def test_load_policy_details_not_found(mock_load):
    from ai_engine.exceptions import PolicyNotFoundError

    mock_load.side_effect = PolicyNotFoundError("policy_missing")
    with pytest.raises(PolicyNotFoundError):
        load_policy_details("policy_missing")


def test_extraction_raises_never_invents_values():
    """The extractor must not fill unknown fields with example values."""
    extractor = PolicyExtractor(llm_provider=EmptyDictLLM())
    details = extractor.extract([PageContent(page_number=1, page_text="Example: room rent ₹5,000.", source_filename="p.pdf")], source_filename="p.pdf")
    assert details.room_rent_limit is None
    assert "room_rent_limit" in details.unknown_fields
