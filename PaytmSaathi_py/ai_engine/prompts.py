"""Prompt templates for the Paytm Saathi RAG engine.

Each function builds the user-facing prompt; the system-level mandatory
instructions are sent separately as the system message so they are kept
distinct from user content.
"""

from __future__ import annotations

import json
from typing import Optional


MANDATORY_RAG_INSTRUCTIONS = """You are Paytm Saathi, a careful health-insurance assistant for claim pre-flight.
MANDATORY RULES - you must follow all of them:
1. Answer ONLY using the supplied policy context. Do not use outside knowledge.
2. Never invent or guess coverage, exclusions, limits, waiting periods, or financial figures.
3. If the policy does not provide enough information, reply exactly with:
   "I could not find enough information in the uploaded policy to answer this reliably."
4. Distinguish clearly between covered, excluded, conditional, and unclear situations.
5. Explain insurance terminology in simple English.
6. Mention all relevant conditions and exceptions.
7. Never guarantee that an insurer will approve a claim.
8. Never calculate claim payouts, deductions, or settlement amounts.
9. Never present an estimate as a guaranteed settlement.
10. Cite sources using [Source N] where N is the number of a context item below.
"""


def _format_context(context_items: list[dict]) -> str:
    lines: list[str] = []
    for item in context_items:
        label = item.get("section_label") or item.get("clause_reference") or "policy clause"
        lines.append(f"[Source {item['index']}] Page {item['page_number']}, {label}:")
        lines.append(f'"{item["text"]}"')
        lines.append("")
    return "\n".join(lines)


def policy_qa_prompt(question: str, context_items: list[dict]) -> str:
    """Prompt for answering a policy question from retrieved clauses."""
    context = _format_context(context_items)
    return f"""{MANDATORY_RAG_INSTRUCTIONS}

POLICY CONTEXT (retrieved clauses from the user's uploaded policy document):
{context}

USER QUESTION: {question}

Answer using only the context above. Cite every factual statement with [Source N].
"""


def follow_up_prompt(
    question: str, context_items: list[dict], conversation_history: list[dict]
) -> str:
    """Prompt for follow-up questions that preserves conversational context."""
    history_text = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}"
        for turn in conversation_history[-6:]
    )
    context = _format_context(context_items)
    return f"""{MANDATORY_RAG_INSTRUCTIONS}

PREVIOUS CONVERSATION (for context only - still ground answers in the policy context below):
{history_text}

POLICY CONTEXT (retrieved clauses from the user's uploaded policy document):
{context}

FOLLOW-UP QUESTION: {question}

Answer using the policy context. Do not contradict the policy. Cite sources with [Source N].
"""


def simplification_prompt(clause_text: str) -> str:
    """Prompt to explain a single clause in simple language."""
    return f"""{MANDATORY_RAG_INSTRUCTIONS}

Explain the following insurance clause in simple, plain English that a
policyholder can easily understand (initially English; Hindi-ready design later).

ORIGINAL CLAUSE:
"{clause_text}"

Rules:
- Preserve the original meaning exactly.
- Do not add any coverage, benefits, or promises not present in the clause.
- Do not calculate payouts or deductibles.
- If the clause is unclear, say so and explain why.
Return only the simple explanation.
"""


def structured_extraction_prompt(document_text: str) -> str:
    """Prompt to extract structured policy information from the full document."""
    return f"""{MANDATORY_RAG_INSTRUCTIONS}

Extract the following policy information from the document below. Return ONLY a
valid JSON object (no markdown fences, no extra prose) with these keys:

{{
  "insurer_name": string | null,
  "policy_name": string | null,
  "policy_type": string | null,
  "sum_insured": {{"value": number|null, "currency": "INR", "source_page": int|null,
                   "source_clause": string|null, "quoted_text": string|null, "is_explicit": bool}},
  "room_rent_limit": {{same shape as sum_insured}},
  "icu_room_rent_limit": {{same shape as sum_insured}},
  "copay_percentage": number|null,
  "deductibles": string|null,
  "waiting_periods": [{{"condition": string, "duration": string|null,
                        "duration_days": int|null, "source_page": int|null,
                        "source_clause": string|null, "quoted_text": string|null}}],
  "pre_existing_disease_waiting_period": {{same entry shape}} | null,
  "initial_waiting_period": {{same entry shape}} | null,
  "specific_disease_waiting_periods": [same entry shape],
  "exclusions": [{{"condition": string, "source_page": int|null,
                   "source_clause": string|null, "quoted_text": string|null}}],
  "sub_limits": [{{"condition": string, "limit": string|null,
                   "source_page": int|null, "source_clause": string|null,
                   "quoted_text": string|null}}],
  "pre_hospitalization_coverage_period": string|null,
  "post_hospitalization_coverage_period": string|null,
  "network_hospital_conditions": string|null,
  "claim_conditions": [{{"condition": string, "source_page": int|null,
                         "source_clause": string|null, "quoted_text": string|null}}],
  "clause_references": [string],
  "unknown_fields": [string]
}}

Rules:
- Use null for any field not found in the document. Never invent values.
- Include source_page and source_clause (e.g. "Section 4.2") wherever visible.
- quoted_text must be the original wording from the document.
- is_explicit: true when the value is directly stated; false when inferred.
- Do not treat examples inside the policy as actual coverage limits.
- Do not convert ambiguous language into definitive financial rules.
- List fields you could not find in unknown_fields.

DOCUMENT (page markers included):
---
{document_text}
---
"""


def claim_clauses_prompt(treatment_details: Optional[dict], context_items: list[dict]) -> str:
    """Prompt to surface claim-relevant clauses for the calculation backend."""
    treatment = (
        json.dumps(treatment_details, indent=2, ensure_ascii=False)
        if treatment_details
        else "not provided"
    )
    context = _format_context(context_items)
    return f"""{MANDATORY_RAG_INSTRUCTIONS}

TREATMENT DETAILS (from the claim pre-flight request, may be partial):
{treatment}

POLICY CONTEXT (retrieved clauses):
{context}

Identify clauses relevant to estimating admissibility of this claim:
coverage, exclusions, sub-limits, room rent limits, co-payment, deductibles,
waiting periods, and claim conditions. Return ONLY a JSON array of objects:
{{"clause_reference": string|null, "section_name": string|null,
  "page_number": int|null, "quoted_text": string, "condition_type":
  "coverage|exclusion|limit|waiting_period|copay|deductible",
  "relevance_to_claim": string}}

Do not calculate any payout amounts.
"""
