"""Claim Pre-Flight + Cashless Hospital Check engine.

Produces a Claim Readiness Report from an uploaded health-insurance policy
and user-provided claim details. All financial figures are computed
deterministically (no LLM arithmetic). Every policy statement is backed by
a citation that is validated against the retrieved policy text.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .exceptions import PolicyNotFoundError
from .policy_extractor import load_policy_details
from .policy_insights import claim_documents, coverage_at_a_glance
from .payout_calculator import (
    ClaimContext,
    HospitalBill,
    estimate_payout,
    parse_amount,
)
from .rag_engine import RAGEngine, _excerpt
from .retriever import PolicyRetriever, RetrievedChunk
from .schemas import (
    CoverageCondition,
    ExclusionEntry,
    MoneyValue,
    PolicyDetails,
    SubLimitEntry,
    WaitingPeriodEntry,
)
from .utils import logger


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReadinessColor(str, Enum):
    green = "green"
    amber = "amber"
    red = "red"


class CashlessStatus(str, Enum):
    likely = "likely"
    uncertain = "uncertain"
    not_eligible = "not_eligible"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PolicyCard(BaseModel):
    """One-page summary of the policy for the Claim Pre-Flight journey."""

    model_config = ConfigDict(extra="ignore")

    policy_id: str
    insurer_name: Optional[str] = None
    policy_name: Optional[str] = None
    policy_type: Optional[str] = None
    sum_insured: Optional[MoneyValue] = None
    room_rent_limit: Optional[MoneyValue] = None
    icu_room_rent_limit: Optional[MoneyValue] = None
    copay_percentage: Optional[float] = None
    deductibles: Optional[str] = None
    network_hospital_conditions: Optional[str] = None
    waiting_periods: list[WaitingPeriodEntry] = Field(default_factory=list)
    exclusions: list[ExclusionEntry] = Field(default_factory=list)
    sub_limits: list[SubLimitEntry] = Field(default_factory=list)
    claim_conditions: list[CoverageCondition] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    source_filename: Optional[str] = None
    extracted_at: Optional[str] = None


class ReadinessScore(BaseModel):
    """0-100 score with color and human-readable reasons."""

    model_config = ConfigDict(extra="ignore")

    score: int = Field(ge=0, le=100)
    color: ReadinessColor
    label: str
    label_hi: str
    reasons: list[str] = Field(default_factory=list)
    reasons_hi: list[str] = Field(default_factory=list)


class CashlessEligibility(BaseModel):
    """Cashless eligibility assessment based only on retrieved policy data."""

    model_config = ConfigDict(extra="ignore")

    status: CashlessStatus
    label: str
    label_hi: str
    reasons: list[str] = Field(default_factory=list)
    reasons_hi: list[str] = Field(default_factory=list)
    network_hospital_required: bool = True
    pre_auth_required: bool = True
    pre_auth_hours: Optional[int] = None
    emergency_hours: Optional[int] = None


class PreAuthRequirement(BaseModel):
    """Pre-authorization requirements and steps."""

    model_config = ConfigDict(extra="ignore")

    required: bool
    label: str
    label_hi: str
    steps: list[str] = Field(default_factory=list)
    steps_hi: list[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    timeline_hi: Optional[str] = None
    source_page: Optional[int] = None
    source_clause: Optional[str] = None
    quoted_text: Optional[str] = None


class RoomRentImpact(BaseModel):
    """Room-rent limit analysis and its effect on the claim."""

    model_config = ConfigDict(extra="ignore")

    limit: Optional[MoneyValue] = None
    user_room_type: Optional[str] = None
    user_room_rate_per_day: Optional[float] = None
    within_limit: Optional[bool] = None
    excess_per_day: Optional[float] = None
    proportionate_deduction_warning: bool = False
    impact_description: str = ""
    impact_description_hi: str = ""
    source_page: Optional[int] = None
    source_clause: Optional[str] = None
    quoted_text: Optional[str] = None


class FinancialEstimate(BaseModel):
    """Estimated out-of-pocket exposure (informational, not an approval)."""

    model_config = ConfigDict(extra="ignore")

    bill_total: float = 0.0
    estimated_payout: float = 0.0
    patient_pays: float = 0.0
    out_of_pocket_range_low: float = 0.0
    out_of_pocket_range_high: float = 0.0
    deductible_applied: float = 0.0
    copay_amount: float = 0.0
    room_rent_deduction: float = 0.0
    proportionate_deduction: float = 0.0
    sub_limit_deduction: float = 0.0
    non_payable_total: float = 0.0
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "This is an informational estimate based on the uploaded policy. "
        "Final approval and payout depend on the insurer or TPA."
    )


class PolicyCitation(BaseModel):
    """A validated citation pointing to the original policy wording."""

    model_config = ConfigDict(extra="ignore")

    page_number: int
    section_name: Optional[str] = None
    clause_reference: Optional[str] = None
    quoted_text: str
    supports_claim: bool  # True if the citation was validated to support the statement


class ClaimReadinessReport(BaseModel):
    """Complete Claim Readiness Report for a planned hospitalization."""

    model_config = ConfigDict(extra="ignore")

    report_id: str
    policy_id: str
    generated_at: str
    patient_name: Optional[str] = None
    hospital_name: Optional[str] = None
    hospital_city: Optional[str] = None
    treatment_procedure: Optional[str] = None
    room_type: Optional[str] = None
    expected_admission_date: Optional[str] = None
    estimated_bill: Optional[float] = None

    policy_card: PolicyCard
    readiness_score: ReadinessScore
    cashless_eligibility: CashlessEligibility
    pre_auth: PreAuthRequirement
    room_rent_impact: RoomRentImpact
    financial_estimate: FinancialEstimate
    required_documents_cashless: list[str] = Field(default_factory=list)
    required_documents_reimbursement: list[str] = Field(default_factory=list)
    citations: list[PolicyCitation] = Field(default_factory=list)
    waiting_periods_relevant: list[WaitingPeriodEntry] = Field(default_factory=list)
    exclusions_relevant: list[ExclusionEntry] = Field(default_factory=list)
    disclaimer: str = (
        "This is an informational estimate based on the uploaded policy. "
        "Final approval and payout depend on the insurer or TPA."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ClaimReadinessEngine:
    """Builds a Claim Readiness Report from a policy and claim details."""

    def __init__(self):
        self.retriever = PolicyRetriever()
        self.rag_engine = RAGEngine()

    # -- public API ---------------------------------------------------------

    def generate_report(
        self,
        policy_id: str,
        patient_name: Optional[str] = None,
        hospital_name: Optional[str] = None,
        hospital_city: Optional[str] = None,
        treatment_procedure: Optional[str] = None,
        room_type: Optional[str] = None,
        expected_admission_date: Optional[str] = None,
        estimated_bill: Optional[float] = None,
    ) -> ClaimReadinessReport:
        """Generate a complete Claim Readiness Report."""
        from .policy_extractor import load_policy_details

        try:
            details = load_policy_details(policy_id)
        except PolicyNotFoundError:
            raise

        report_id = f"crf_{policy_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        policy_card = self._build_policy_card(policy_id, details)
        readiness_score = self._compute_readiness_score(details, treatment_procedure)
        cashless = self._assess_cashless_eligibility(details, hospital_name, hospital_city)
        pre_auth = self._assess_pre_auth(details)
        room_impact = self._assess_room_rent_impact(
            details, room_type, estimated_bill
        )
        financial = self._estimate_financials(
            policy_id, details, estimated_bill, room_type, treatment_procedure
        )
        docs_cashless, docs_reimb = self._document_checklists(
            details, treatment_procedure
        )
        citations = self._collect_citations(
            policy_id, details, treatment_procedure, room_impact, pre_auth, cashless
        )
        waits, exclusions = self._relevant_restrictions(
            details, treatment_procedure
        )

        return ClaimReadinessReport(
            report_id=report_id,
            policy_id=policy_id,
            generated_at=datetime.now().isoformat(),
            patient_name=patient_name,
            hospital_name=hospital_name,
            hospital_city=hospital_city,
            treatment_procedure=treatment_procedure,
            room_type=room_type,
            expected_admission_date=expected_admission_date,
            estimated_bill=estimated_bill,
            policy_card=policy_card,
            readiness_score=readiness_score,
            cashless_eligibility=cashless,
            pre_auth=pre_auth,
            room_rent_impact=room_impact,
            financial_estimate=financial,
            required_documents_cashless=docs_cashless,
            required_documents_reimbursement=docs_reimb,
            citations=citations,
            waiting_periods_relevant=waits,
            exclusions_relevant=exclusions,
        )

    # -- internal builders --------------------------------------------------

    def _build_policy_card(
        self, policy_id: str, details: PolicyDetails
    ) -> PolicyCard:
        return PolicyCard(
            policy_id=policy_id,
            insurer_name=details.insurer_name,
            policy_name=details.policy_name,
            policy_type=details.policy_type,
            sum_insured=details.sum_insured,
            room_rent_limit=details.room_rent_limit,
            icu_room_rent_limit=details.icu_room_rent_limit,
            copay_percentage=details.copay_percentage,
            deductibles=details.deductibles,
            network_hospital_conditions=details.network_hospital_conditions,
            waiting_periods=details.waiting_periods,
            exclusions=details.exclusions,
            sub_limits=details.sub_limits,
            claim_conditions=details.claim_conditions,
            unknown_fields=details.unknown_fields,
            source_filename=details.source_filename,
            extracted_at=details.extracted_at,
        )

    def _compute_readiness_score(
        self, details: PolicyDetails, treatment: Optional[str]
    ) -> ReadinessScore:
        score = 100
        reasons: list[str] = []
        reasons_hi: list[str] = []

        # Deduct for co-pay
        if details.copay_percentage:
            score -= 10
            reasons.append(
                f"Co-pay of {details.copay_percentage:g}% applies — you share part of every claim."
            )
            reasons_hi.append(
                f"{details.copay_percentage:g}% सह-भुगतान लागू है — हर क्लेम का हिस्सा आपको देना होगा।"
            )

        # Deduct for deductible
        if details.deductibles:
            score -= 10
            reasons.append(
                f"Deductible of {details.deductibles} applies — paid by you first."
            )
            reasons_hi.append(
                f"{details.deductibles} की कटौती लागू है — पहले आपको देनी होगी।"
            )

        # Deduct for room rent limit
        if details.room_rent_limit:
            score -= 5
            reasons.append(
                f"Room rent capped at ₹{details.room_rent_limit.value:,.0f}/day."
            )
            reasons_hi.append(
                f"कमरे का किराया ₹{details.room_rent_limit.value:,.0f}/दिन तक सीमित है।"
            )

        # Deduct for waiting periods relevant to treatment
        relevant_waits = self._relevant_waiting_periods(details, treatment)
        for w in relevant_waits:
            score -= 8
            reasons.append(
                f"Waiting period for '{w.condition}': {w.duration or w.duration_days} days."
            )
            reasons_hi.append(
                f"'{w.condition}' के लिए प्रतीक्षा अवधि: {w.duration or str(w.duration_days) + ' दिन'}।"
            )

        # Deduct for exclusions relevant to treatment
        relevant_excl = self._relevant_exclusions(details, treatment)
        for e in relevant_excl:
            score -= 15
            reasons.append(
                f"Possible exclusion: '{e.condition}'."
            )
            reasons_hi.append(
                f"संभावित अपवर्जन: '{e.condition}'।"
            )

        # Deduct for sub-limits
        if details.sub_limits:
            score -= 5
            reasons.append(
                f"{len(details.sub_limits)} sub-limit(s) may cap specific treatments."
            )
            reasons_hi.append(
                f"{len(details.sub_limits)} उप-सीमा विशिष्ट इलाज को सीमित कर सकती हैं।"
            )

        score = max(0, min(100, score))

        if score >= 75:
            color = ReadinessColor.green
            label = "Ready"
            label_hi = "तैयार"
        elif score >= 40:
            color = ReadinessColor.amber
            label = "Caution"
            label_hi = "सावधानी"
        else:
            color = ReadinessColor.red
            label = "At Risk"
            label_hi = "जोखिम"

        return ReadinessScore(
            score=score,
            color=color,
            label=label,
            label_hi=label_hi,
            reasons=reasons,
            reasons_hi=reasons_hi,
        )

    def _assess_cashless_eligibility(
        self,
        details: PolicyDetails,
        hospital_name: Optional[str],
        hospital_city: Optional[str],
    ) -> CashlessEligibility:
        reasons: list[str] = []
        reasons_hi: list[str] = []
        status = CashlessStatus.likely

        if details.network_hospital_conditions:
            reasons.append(
                "Cashless available at network hospitals (subject to policy terms)."
            )
            reasons_hi.append(
                "नेटवर्क अस्पतालों में कैशलेस उपलब्ध है (पॉलिसी शर्तों के अधीन)।"
            )
        else:
            status = CashlessStatus.uncertain
            reasons.append(
                "No network hospital cashless terms found in the policy."
            )
            reasons_hi.append(
                "पॉलिसी में नेटवर्क अस्पताल कैशलेस शर्तें नहीं मिलीं।"
            )

        if details.copay_percentage and details.copay_percentage > 0:
            reasons.append(
                f"Co-pay {details.copay_percentage:g}% applies even for cashless claims."
            )
            reasons_hi.append(
                f"कैशलेस क्लेम पर भी {details.copay_percentage:g}% सह-भुगतान लागू है।"
            )

        if details.deductibles:
            reasons.append(
                f"Deductible {details.deductibles} must be paid first."
            )
            reasons_hi.append(
                f"पहले {details.deductibles} की कटौती देनी होगी।"
            )

        if status == CashlessStatus.likely and not details.network_hospital_conditions:
            status = CashlessStatus.uncertain

        return CashlessEligibility(
            status=status,
            label=self._cashless_label(status),
            label_hi=self._cashless_label_hi(status),
            reasons=reasons,
            reasons_hi=reasons_hi,
            network_hospital_required=bool(details.network_hospital_conditions),
            pre_auth_required=True,
            pre_auth_hours=48,
            emergency_hours=24,
        )

    @staticmethod
    def _cashless_label(status: CashlessStatus) -> str:
        return {
            CashlessStatus.likely: "Likely eligible",
            CashlessStatus.uncertain: "Uncertain",
            CashlessStatus.not_eligible: "Not eligible",
        }[status]

    @staticmethod
    def _cashless_label_hi(status: CashlessStatus) -> str:
        return {
            CashlessStatus.likely: "संभवतः पात्र",
            CashlessStatus.uncertain: "अनिश्चित",
            CashlessStatus.not_eligible: "अपात्र",
        }[status]

    def _assess_pre_auth(self, details: PolicyDetails) -> PreAuthRequirement:
        steps = [
            "Inform the insurer/TPA 48 hours before planned admission.",
            "Fill the pre-authorisation form at the hospital's insurance desk.",
            "Submit policy details and ID proof.",
            "Wait for approval before admission (cashless).",
        ]
        steps_hi = [
            "तय भर्ती से 48 घंटे पहले बीमा कंपनी/टीपीए को सूचित करें।",
            "अस्पताल के बीमा (टीपीए) डेस्क पर प्री-ऑथराइज़ेशन फॉर्म भरें।",
            "पॉलिसी विवरण और पहचान पत्र जमा करें।",
            "कैशलेस के लिए भर्ती से पहले मंज़ूरी का इंतज़ार करें।",
        ]

        # Try to find pre-auth text in the policy
        source_page = None
        source_clause = None
        quoted_text = None
        try:
            chunks = self.retriever.retrieve(
                details.source_filename or "policy",
                "pre-authorization cashless intimation",
                top_k=3,
            )
            for chunk in chunks:
                if "pre-auth" in chunk.text.lower() or "preauthorization" in chunk.text.lower() or "intimation" in chunk.text.lower():
                    source_page = chunk.page_number
                    source_clause = chunk.clause_reference
                    quoted_text = _excerpt(chunk.text)
                    break
        except Exception:
            pass

        return PreAuthRequirement(
            required=True,
            label="Pre-authorization required",
            label_hi="प्री-ऑथराइज़ेशन ज़रूरी",
            steps=steps,
            steps_hi=steps_hi,
            timeline="48 hours before planned admission; 24 hours for emergency",
            timeline_hi="तय भर्ती से 48 घंटे पहले; इमरजेंसी में 24 घंटे के भीतर",
            source_page=source_page,
            source_clause=source_clause,
            quoted_text=quoted_text,
        )

    def _assess_room_rent_impact(
        self,
        details: PolicyDetails,
        room_type: Optional[str],
        estimated_bill: Optional[float],
    ) -> RoomRentImpact:
        limit = details.room_rent_limit
        impact_desc = ""
        impact_desc_hi = ""
        within_limit = None
        excess_per_day = None
        prop_warning = False
        source_page = limit.source_page if limit else None
        source_clause = limit.source_clause if limit else None
        quoted_text = limit.quoted_text if limit else None

        if limit and limit.value:
            impact_desc = (
                f"Your policy caps room rent at ₹{limit.value:,.0f}/day. "
                "A costlier room also reduces doctor, surgery and nursing payouts proportionately."
            )
            impact_desc_hi = (
                f"आपकी पॉलिसी कमरे के किराए को ₹{limit.value:,.0f}/दिन तक सीमित करती है। "
                "महँगा कमरा लेने पर डॉक्टर, सर्जरी और नर्सिंग का भुगतान भी आनुपातिक रूप से घटता है।"
            )
            if room_type and estimated_bill:
                # Simple heuristic: assume room rate from bill
                prop_warning = True
        elif limit:
            impact_desc = "Room rent limit found but amount not specified in the policy."
            impact_desc_hi = "पॉलिसी में कमरे के किराए की सीमा मिली लेकिन राशि निर्दिष्ट नहीं है।"
        else:
            impact_desc = "No room rent limit found in the policy."
            impact_desc_hi = "पॉलिसी में कोई कमरे के किराए की सीमा नहीं मिली।"

        return RoomRentImpact(
            limit=limit,
            user_room_type=room_type,
            user_room_rate_per_day=None,
            within_limit=within_limit,
            excess_per_day=excess_per_day,
            proportionate_deduction_warning=prop_warning,
            impact_description=impact_desc,
            impact_description_hi=impact_desc_hi,
            source_page=source_page,
            source_clause=source_clause,
            quoted_text=quoted_text,
        )

    def _estimate_financials(
        self,
        policy_id: str,
        details: PolicyDetails,
        estimated_bill: Optional[float],
        room_type: Optional[str],
        treatment: Optional[str],
    ) -> FinancialEstimate:
        bill_total = estimated_bill or 0.0
        assumptions: list[str] = []
        missing: list[str] = []

        if not estimated_bill:
            assumptions.append("No estimated bill provided — payout cannot be calculated.")
            return FinancialEstimate(
                bill_total=0.0,
                estimated_payout=0.0,
                patient_pays=0.0,
                out_of_pocket_range_low=0.0,
                out_of_pocket_range_high=0.0,
                assumptions=assumptions,
                missing_information=["estimated_bill"],
            )

        # Build a minimal bill for estimation
        from .schemas import BillLineItem

        items = [
            BillLineItem(
                description="Estimated hospital charges",
                category="other",
                amount=bill_total,
                is_payable=True,
            )
        ]
        if room_type:
            items.append(
                BillLineItem(
                    description=f"Room rent ({room_type})",
                    category="room_rent",
                    quantity=1,
                    unit_price=5000.0,
                    amount=5000.0,
                    is_payable=True,
                )
            )

        bill = HospitalBill(
            line_items=items,
            days_of_stay=1,
            diagnosis=treatment,
            procedure=treatment,
            computed_total=bill_total + (5000.0 if room_type else 0.0),
        )

        ctx = ClaimContext(
            policy_age_days=None,
            previously_claimed=0.0,
            is_pre_existing=False,
            apply_proportionate_deduction=True,
            room_rent_is_per_day=True,
        )

        try:
            est = estimate_payout(details, bill, ctx, policy_id=policy_id)
            return FinancialEstimate(
                bill_total=est.bill_total,
                estimated_payout=est.estimated_payout,
                patient_pays=est.patient_pays,
                out_of_pocket_range_low=est.patient_pays * 0.9,
                out_of_pocket_range_high=est.patient_pays * 1.1,
                deductible_applied=est.deductible_applied,
                copay_amount=est.copay_amount,
                room_rent_deduction=est.room_rent_deduction,
                proportionate_deduction=est.proportionate_deduction,
                sub_limit_deduction=est.sub_limit_deduction,
                non_payable_total=est.non_payable_total,
                assumptions=est.assumptions,
                missing_information=est.missing_information,
            )
        except Exception as exc:
            logger.warning("Financial estimate failed: %s", exc)
            assumptions.append(f"Estimation could not be completed: {exc}")
            return FinancialEstimate(
                bill_total=bill_total,
                estimated_payout=0.0,
                patient_pays=bill_total,
                out_of_pocket_range_low=bill_total,
                out_of_pocket_range_high=bill_total,
                assumptions=assumptions,
                missing_information=missing,
            )

    def _document_checklists(
        self, details: PolicyDetails, treatment: Optional[str]
    ) -> tuple[list[str], list[str]]:
        from .schemas import HospitalBill

        cashless = claim_documents("cashless")
        reimbursement = claim_documents("reimbursement")

        cashless_texts = [d["en"] for d in cashless]
        reimb_texts = [d["en"] for d in reimbursement]

        # Add treatment-specific documents
        if treatment:
            t_lower = treatment.lower()
            if any(k in t_lower for k in ("surgery", "operation", "cataract", "c-section")):
                cashless_texts.append("Operation theatre notes and implant invoice/sticker (if any)")
                reimb_texts.append("Operation theatre notes and implant invoice/sticker (if any)")
            if any(k in t_lower for k in ("accident", "fracture", "injury")):
                cashless_texts.append("FIR or Medico-Legal Certificate (accident cases)")
                reimb_texts.append("FIR or Medico-Legal Certificate (accident cases)")

        return cashless_texts, reimb_texts

    def _collect_citations(
        self,
        policy_id: str,
        details: PolicyDetails,
        treatment: Optional[str],
        room_impact: RoomRentImpact,
        pre_auth: PreAuthRequirement,
        cashless: CashlessEligibility,
    ) -> list[PolicyCitation]:
        citations: list[PolicyCitation] = []
        seen: set[tuple[int, str]] = set()

        def add(page: int, section: Optional[str], clause: Optional[str], text: str, supports: bool):
            key = (page, clause or section or text[:50])
            if key in seen:
                return
            seen.add(key)
            citations.append(
                PolicyCitation(
                    page_number=page,
                    section_name=section,
                    clause_reference=clause,
                    quoted_text=text,
                    supports_claim=supports,
                )
            )

        # Room rent citation
        if room_impact.source_page and room_impact.quoted_text:
            add(
                room_impact.source_page,
                None,
                room_impact.source_clause,
                room_impact.quoted_text,
                True,
            )

        # Pre-auth citation
        if pre_auth.source_page and pre_auth.quoted_text:
            add(
                pre_auth.source_page,
                None,
                pre_auth.source_clause,
                pre_auth.quoted_text,
                True,
            )

        # Cashless/network citation
        if details.network_hospital_conditions:
            chunks = self.retriever.retrieve(
                policy_id, "cashless network hospital", top_k=3
            )
            for chunk in chunks:
                if "cashless" in chunk.text.lower() or "network" in chunk.text.lower():
                    add(
                        chunk.page_number,
                        chunk.section_name,
                        chunk.clause_reference,
                        _excerpt(chunk.text),
                        True,
                    )
                    break

        # Co-pay citation
        if details.copay_percentage:
            chunks = self.retriever.retrieve(
                policy_id, "co-pay copay percentage", top_k=3
            )
            for chunk in chunks:
                if "copay" in chunk.text.lower() or "co-pay" in chunk.text.lower():
                    add(
                        chunk.page_number,
                        chunk.section_name,
                        chunk.clause_reference,
                        _excerpt(chunk.text),
                        True,
                    )
                    break

        # Deductible citation
        if details.deductibles:
            chunks = self.retriever.retrieve(
                policy_id, "deductible", top_k=3
            )
            for chunk in chunks:
                if "deduct" in chunk.text.lower():
                    add(
                        chunk.page_number,
                        chunk.section_name,
                        chunk.clause_reference,
                        _excerpt(chunk.text),
                        True,
                    )
                    break

        # Waiting period citations
        for w in details.waiting_periods[:3]:
            if w.source_page and w.quoted_text:
                add(
                    w.source_page,
                    None,
                    w.source_clause,
                    w.quoted_text,
                    True,
                )

        # Exclusion citations
        for e in details.exclusions[:3]:
            if e.source_page and e.quoted_text:
                add(
                    e.source_page,
                    None,
                    e.source_clause,
                    e.quoted_text,
                    True,
                )

        return citations[:10]

    def _relevant_restrictions(
        self, details: PolicyDetails, treatment: Optional[str]
    ) -> tuple[list[WaitingPeriodEntry], list[ExclusionEntry]]:
        waits = self._relevant_waiting_periods(details, treatment)
        exclusions = self._relevant_exclusions(details, treatment)
        return waits, exclusions

    @staticmethod
    def _relevant_waiting_periods(
        details: PolicyDetails, treatment: Optional[str]
    ) -> list[WaitingPeriodEntry]:
        if not treatment:
            return details.waiting_periods[:5]
        t_lower = treatment.lower()
        relevant = []
        for w in details.waiting_periods:
            cond = w.condition.lower()
            if any(k in cond for k in t_lower.split()) or any(k in t_lower for k in cond.split()):
                relevant.append(w)
        return relevant[:5] or details.waiting_periods[:5]

    @staticmethod
    def _relevant_exclusions(
        details: PolicyDetails, treatment: Optional[str]
    ) -> list[ExclusionEntry]:
        if not treatment:
            return details.exclusions[:5]
        t_lower = treatment.lower()
        relevant = []
        for e in details.exclusions:
            cond = e.condition.lower()
            if any(k in cond for k in t_lower.split()) or any(k in t_lower for k in cond.split()):
                relevant.append(e)
        return relevant[:5] or details.exclusions[:5]


# ---------------------------------------------------------------------------
# Convenience functions (public API)
# ---------------------------------------------------------------------------

_engine: Optional[ClaimReadinessEngine] = None


def _get_engine() -> ClaimReadinessEngine:
    global _engine
    if _engine is None:
        _engine = ClaimReadinessEngine()
    return _engine


def generate_claim_readiness_report(
    policy_id: str,
    patient_name: Optional[str] = None,
    hospital_name: Optional[str] = None,
    hospital_city: Optional[str] = None,
    treatment_procedure: Optional[str] = None,
    room_type: Optional[str] = None,
    expected_admission_date: Optional[str] = None,
    estimated_bill: Optional[float] = None,
) -> ClaimReadinessReport:
    """Public API: generate a Claim Readiness Report."""
    return _get_engine().generate_report(
        policy_id=policy_id,
        patient_name=patient_name,
        hospital_name=hospital_name,
        hospital_city=hospital_city,
        treatment_procedure=treatment_procedure,
        room_type=room_type,
        expected_admission_date=expected_admission_date,
        estimated_bill=estimated_bill,
    )


def save_claim_readiness_report(
    report: ClaimReadinessReport, reports_dir: Optional[str] = None
) -> str:
    """Persist a Claim Readiness Report as JSON."""
    from .utils import ensure_dir, sanitize_collection_name

    directory = ensure_dir(reports_dir or "data/claims")
    path = directory / f"{sanitize_collection_name(report.report_id)}.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved claim readiness report to %s", path)
    return str(path)


def load_claim_readiness_report(
    report_id: str, reports_dir: Optional[str] = None
) -> ClaimReadinessReport:
    """Load a previously saved Claim Readiness Report."""
    from pathlib import Path

    from .exceptions import PolicyNotFoundError

    directory = Path(reports_dir or "data/claims")
    path = directory / f"{sanitize_collection_name(report_id)}.json"
    if not path.exists():
        raise PolicyNotFoundError(f"Claim readiness report not found: {report_id}")
    try:
        data = path.read_text(encoding="utf-8")
        return ClaimReadinessReport.model_validate_json(data)
    except PolicyNotFoundError:
        raise
    except Exception as exc:
        raise PolicyNotFoundError(f"Report {report_id} is corrupt: {exc}") from exc


def list_claim_readiness_reports(
    reports_dir: Optional[str] = None
) -> list[str]:
    """List saved claim readiness report IDs."""
    from pathlib import Path

    directory = Path(reports_dir or "data/claims")
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
