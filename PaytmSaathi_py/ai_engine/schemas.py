"""Pydantic schemas for the Paytm Saathi AI engine.

Every structured input and output of the AI engine is validated with these
models. LLM-generated data is validated with Pydantic before it is trusted
anywhere else; malformed data is rejected safely, never silently coerced.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceLocation(BaseModel):
    """Where a piece of policy text appears in the source PDF."""

    model_config = ConfigDict(extra="ignore")

    page_number: int
    section_name: Optional[str] = None
    clause_reference: Optional[str] = None


class MoneyValue(BaseModel):
    """A monetary amount as stated in a policy document.

    `is_explicit` distinguishes a value written directly in the document from
    a value inferred from context. Inferred values must never be treated as
    definitive financial rules by the calculation backend.
    """

    model_config = ConfigDict(extra="ignore")

    value: Optional[float] = None
    currency: Optional[str] = "INR"
    source_page: Optional[int] = None
    source_clause: Optional[str] = None
    quoted_text: Optional[str] = None
    is_explicit: bool = True

    @field_validator("value", mode="before")
    @classmethod
    def _clean_number(cls, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace("₹", "").replace(",", "").replace(" ", "").strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return value


class PageContent(BaseModel):
    """One page extracted from a policy PDF."""

    model_config = ConfigDict(extra="ignore")

    page_number: int
    page_text: str
    source_filename: str


class DocumentExtractionResult(BaseModel):
    """Result of extracting a policy PDF."""

    model_config = ConfigDict(extra="ignore")

    source_filename: str
    total_pages: int
    pages: list[PageContent] = Field(default_factory=list)
    has_extractable_text: bool = False
    is_scanned: bool = False


class Chunk(BaseModel):
    """One text chunk with full provenance metadata."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    text: str
    page_number: int
    section_name: Optional[str] = None
    clause_reference: Optional[str] = None
    source_filename: str


class WaitingPeriodEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    condition: str
    duration: Optional[str] = None
    duration_days: Optional[int] = None
    source_page: Optional[int] = None
    source_clause: Optional[str] = None
    quoted_text: Optional[str] = None

    @field_validator("duration_days", mode="before")
    @classmethod
    def _clean_days(cls, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"(\d+)\s*(day|days)?", value, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return None
        return value


class ExclusionEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    condition: str
    source_page: Optional[int] = None
    source_clause: Optional[str] = None
    quoted_text: Optional[str] = None


class SubLimitEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    condition: str
    limit: Optional[str] = None
    source_page: Optional[int] = None
    source_clause: Optional[str] = None
    quoted_text: Optional[str] = None


class CoverageCondition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    condition: str
    source_page: Optional[int] = None
    source_clause: Optional[str] = None
    quoted_text: Optional[str] = None


class PolicyDetails(BaseModel):
    """Structured information extracted from a health insurance policy.

    Missing fields are `None` / empty lists and listed in `unknown_fields`.
    Never assume a value that is not in the document.
    """

    model_config = ConfigDict(extra="ignore")

    insurer_name: Optional[str] = None
    policy_name: Optional[str] = None
    policy_type: Optional[str] = None
    sum_insured: Optional[MoneyValue] = None
    room_rent_limit: Optional[MoneyValue] = None
    icu_room_rent_limit: Optional[MoneyValue] = None
    copay_percentage: Optional[float] = None
    deductibles: Optional[str] = None
    waiting_periods: list[WaitingPeriodEntry] = Field(default_factory=list)
    pre_existing_disease_waiting_period: Optional[WaitingPeriodEntry] = None
    initial_waiting_period: Optional[WaitingPeriodEntry] = None
    specific_disease_waiting_periods: list[WaitingPeriodEntry] = (
        Field(default_factory=list)
    )
    exclusions: list[ExclusionEntry] = Field(default_factory=list)
    sub_limits: list[SubLimitEntry] = Field(default_factory=list)
    pre_hospitalization_coverage_period: Optional[str] = None
    post_hospitalization_coverage_period: Optional[str] = None
    network_hospital_conditions: Optional[str] = None
    claim_conditions: list[CoverageCondition] = Field(default_factory=list)
    clause_references: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    extracted_at: Optional[str] = None
    source_filename: Optional[str] = None

    @field_validator("copay_percentage", mode="before")
    @classmethod
    def _clean_copay(cls, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.replace("%", "").replace("₹", "").replace(",", "").strip()
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return value


class SourceCitation(BaseModel):
    """A structured citation pointing back to the original policy wording."""

    model_config = ConfigDict(extra="ignore")

    page_number: int
    section_name: Optional[str] = None
    clause_reference: Optional[str] = None
    quoted_text: str


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class RAGResponse(BaseModel):
    """Standard response format for policy question answering."""

    model_config = ConfigDict(extra="ignore")

    policy_id: str
    question: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.low
    insufficient_evidence: bool = False
    conversation_id: Optional[str] = None
    language: str = "en"  # language of `answer` ("en" or "hi")
    answer_en: Optional[str] = None  # English original when `answer` was translated


class SimpleExplanation(BaseModel):
    """Original clause plus a plain-language explanation."""

    model_config = ConfigDict(extra="ignore")

    policy_id: str
    clause_reference: Optional[str] = None
    page_number: Optional[int] = None
    original_clause: str
    simple_explanation: str
    language: str = "en"  # language of `simple_explanation`
    source_filename: Optional[str] = None


class ClaimRelevantClause(BaseModel):
    """One clause relevant to a claim, categorized for the calculation backend."""

    model_config = ConfigDict(extra="ignore")

    clause_reference: Optional[str] = None
    section_name: Optional[str] = None
    page_number: Optional[int] = None
    quoted_text: str
    condition_type: str  # coverage | exclusion | limit | waiting_period | copay | deductible
    relevance_to_claim: str


class ClaimRelevantClausesResult(BaseModel):
    """Output of get_claim_relevant_clauses for the calculation backend."""

    model_config = ConfigDict(extra="ignore")

    policy_id: str
    treatment_details: Optional[dict] = None
    clauses: list[ClaimRelevantClause] = Field(default_factory=list)
    structured_conditions: dict = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)



# ---------------------------------------------------------------------------
# Hospital bill + payout estimate models
# ---------------------------------------------------------------------------
class BillLineItem(BaseModel):
    """One charge on a hospital bill."""

    model_config = ConfigDict(extra="ignore")

    description: str
    category: str = "other"  # room_rent | icu_rent | doctor_fees | surgery | nursing |
    #                          medicines | investigations | consumables | non_payable | other
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: float = 0.0
    is_payable: bool = True
    note: Optional[str] = None

    @field_validator("amount", "quantity", "unit_price", mode="before")
    @classmethod
    def _clean_number(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.\-]", "", value)
            try:
                return float(cleaned) if cleaned not in ("", "-", ".") else None
            except ValueError:
                return None
        return value

    @field_validator("amount", mode="after")
    @classmethod
    def _amount_not_none(cls, value):
        return 0.0 if value is None else value


class HospitalBill(BaseModel):
    """Structured hospital bill. Missing fields stay `None`, never guessed."""

    model_config = ConfigDict(extra="ignore")

    hospital_name: Optional[str] = None
    patient_name: Optional[str] = None
    bill_number: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    days_of_stay: Optional[int] = None
    diagnosis: Optional[str] = None
    procedure: Optional[str] = None
    room_type: Optional[str] = None
    is_network_hospital: Optional[bool] = None
    line_items: list[BillLineItem] = Field(default_factory=list)
    stated_total: Optional[float] = None
    computed_total: float = 0.0
    extraction_method: str = "heuristic"  # heuristic | llm | manual
    used_ocr: bool = False
    warnings: list[str] = Field(default_factory=list)
    source_filename: Optional[str] = None
    language: str = "en"

    @field_validator("stated_total", mode="before")
    @classmethod
    def _clean_total(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.]", "", value)
            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                return None
        return value


class ClaimContext(BaseModel):
    """Facts about the claim that are not on the bill."""

    model_config = ConfigDict(extra="ignore")

    policy_age_days: Optional[int] = None  # days since policy start (waiting periods)
    previously_claimed: float = 0.0  # already used from the sum insured this year
    is_pre_existing: bool = False
    apply_proportionate_deduction: bool = True
    room_rent_is_per_day: bool = True


class PayoutStep(BaseModel):
    """One transparent step of the estimate, with the clause it relies on."""

    model_config = ConfigDict(extra="ignore")

    step: str
    description: str
    description_hi: Optional[str] = None
    amount_before: float
    amount_after: float
    deduction: float = 0.0
    clause_reference: Optional[str] = None
    source_page: Optional[int] = None


class PayoutEstimate(BaseModel):
    """Deterministic claim payout estimate. Never a guarantee of approval."""

    model_config = ConfigDict(extra="ignore")

    policy_id: Optional[str] = None
    bill_total: float
    non_payable_total: float = 0.0
    room_rent_deduction: float = 0.0
    proportionate_deduction: float = 0.0
    sub_limit_deduction: float = 0.0
    deductible_applied: float = 0.0
    copay_amount: float = 0.0
    sum_insured_cap_deduction: float = 0.0
    payout_if_covered: float = 0.0
    estimated_payout: float = 0.0
    patient_pays: float = 0.0
    eligibility_status: str = "likely_payable"  # likely_payable | needs_review | likely_excluded
    steps: list[PayoutStep] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    flags_hi: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an estimate based on the policy details extracted by AI and the bill "
        "provided. It is not a claim approval or a guarantee of payment. The insurer's "
        "final decision applies."
    )
