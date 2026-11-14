"""Paytm Saathi REST API (FastAPI).

Run from the project root:

    uvicorn backend.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ai_engine import (
    ClaimContext,
    ClaimRelevantClausesResult,
    HospitalBill,
    PayoutEstimate,
    PolicyDetails,
    RAGResponse,
    SimpleExplanation,
    ask_policy_question,
    estimate_claim_payout,
    explain_policy_clause,
    extract_hospital_bill,
    get_claim_relevant_clauses,
    get_policy_details,
    list_policy_ids,
    process_policy,
    translate_text,
)
from ai_engine import accounts, fintech, lending, policy_insights
from ai_engine import explain_clause_text
from ai_engine.exceptions import PaytmSaathiError
from ai_engine.ocr import available_backend

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(
    title="Paytm Saathi API",
    version="1.0.0",
    description="AI-powered help for Insurance, Lending and Fintech journeys.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATUS = {
    "invalid_file": 400, "invalid_question": 400, "empty_pdf": 422, "scanned_pdf": 422,
    "pdf_extraction_error": 422, "validation_error": 422, "bill_extraction_error": 422,
    "calculation_input_error": 422, "policy_not_found": 404, "ticket_not_found": 404,
    "ocr_unavailable": 501, "llm_provider_error": 502, "embedding_error": 500,
    "vector_store_error": 500,
    "invalid_name": 422, "invalid_mobile": 422, "invalid_email": 422, "weak_password": 422,
    "password_mismatch": 422, "terms_not_accepted": 422, "missing_credentials": 400,
    "mobile_taken": 409, "email_taken": 409, "invalid_credentials": 401, "invalid_token": 401,
    "account_locked": 429, "server_misconfigured": 500, "account_storage_error": 500,
    "attachment_not_found": 404, "evidence_rejected": 422,
}


@app.exception_handler(PaytmSaathiError)
async def engine_error_handler(_: Request, exc: PaytmSaathiError):
    return JSONResponse(
        status_code=_STATUS.get(exc.error_code, 400),
        content={"error": exc.error_code, "message": exc.message,
                 **({"message_hi": exc.message_hi} if getattr(exc, "message_hi", None) else {}),
                 **({"reasons": exc.reasons} if getattr(exc, "reasons", None) else {})},
    )


def current_user_id(authorization: str = Header("", description="Bearer <token>")) -> str:
    """Dependency: the logged-in user's id from the signed token."""
    token = authorization.removeprefix("Bearer ").strip()
    return accounts.verify_token(token)


def own_ticket(ticket_id: str, user_id: str) -> "fintech.DisputeTicket":
    """Load a ticket only if it belongs to the caller (others get the same 'not found')."""
    ticket = fintech.get_dispute_ticket(ticket_id)
    if ticket.user_id != user_id:
        raise PaytmSaathiError(f"Ticket not found: {ticket_id}", "ticket_not_found")
    return ticket


async def _save_upload(upload: UploadFile, directory: Path) -> Path:
    """Save an upload under a sanitized name, enforcing the size limit."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(upload.filename or "upload").name)[:100] or "upload"
    target = directory / name
    size = 0
    with target.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File is too large (limit 15 MB).")
            handle.write(chunk)
    return target


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "ocr_backend": available_backend()}


# ---------------------------------------------------------------------------
# Insurance
# ---------------------------------------------------------------------------
@app.get("/policies", tags=["insurance"])
def policies():
    return {"policy_ids": list_policy_ids()}


@app.post("/policies/{policy_id}/ingest", response_model=PolicyDetails, tags=["insurance"])
async def ingest_policy(policy_id: str, file: UploadFile = File(...)):
    """Upload a policy PDF (text or scanned). Returns extracted structured details."""
    with tempfile.TemporaryDirectory() as tmp:
        path = await _save_upload(file, Path(tmp))
        return await run_in_threadpool(process_policy, str(path), policy_id)


@app.get("/policies/{policy_id}/details", response_model=PolicyDetails, tags=["insurance"])
def policy_details(policy_id: str):
    return get_policy_details(policy_id)


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    language: Optional[str] = Field(None, description="'en' or 'hi'; auto-detected if omitted")


@app.post("/policies/{policy_id}/ask", response_model=RAGResponse, tags=["insurance"])
def ask(policy_id: str, body: AskRequest):
    return ask_policy_question(policy_id, body.question, body.conversation_id, body.language)


class ExplainRequest(BaseModel):
    clause_reference: str
    language: str = "en"


@app.post("/policies/{policy_id}/explain", response_model=SimpleExplanation, tags=["insurance"])
def explain(policy_id: str, body: ExplainRequest):
    return explain_policy_clause(policy_id, body.clause_reference, body.language)


class ClaimClausesRequest(BaseModel):
    treatment_details: Optional[dict] = None


@app.post("/policies/{policy_id}/claim-clauses", response_model=ClaimRelevantClausesResult, tags=["insurance"])
def claim_clauses(policy_id: str, body: ClaimClausesRequest):
    return get_claim_relevant_clauses(policy_id, body.treatment_details)


@app.post("/bills/extract", response_model=HospitalBill, tags=["insurance"])
async def extract_bill_endpoint(file: UploadFile = File(...)):
    """Upload a hospital bill (PDF, scan or photo) and get structured line items."""
    with tempfile.TemporaryDirectory() as tmp:
        path = await _save_upload(file, Path(tmp))
        return await run_in_threadpool(extract_hospital_bill, str(path))


class EstimateRequest(BaseModel):
    bill: HospitalBill
    context: Optional[ClaimContext] = None


@app.post("/policies/{policy_id}/estimate", response_model=PayoutEstimate, tags=["insurance"])
def estimate(policy_id: str, body: EstimateRequest):
    """Deterministic payout estimate (editable bill in, transparent steps out)."""
    return estimate_claim_payout(policy_id, body.bill, body.context)


# ---------------------------------------------------------------------------
# Lending
# ---------------------------------------------------------------------------
class EMIRequest(BaseModel):
    principal: float
    annual_rate_pct: float
    tenure_months: int


@app.post("/lending/emi", response_model=lending.LoanSummary, tags=["lending"])
def emi(body: EMIRequest):
    return lending.loan_summary(body.principal, body.annual_rate_pct, body.tenure_months)


@app.post("/lending/schedule", response_model=list[lending.ScheduleRow], tags=["lending"])
def schedule(body: EMIRequest):
    return lending.amortization_schedule(body.principal, body.annual_rate_pct, body.tenure_months)


@app.post("/lending/eligibility", response_model=lending.EligibilityResult, tags=["lending"])
def eligibility(body: lending.LoanApplicant):
    return lending.check_loan_eligibility(body)


@app.get("/lending/documents", tags=["lending"])
def documents(loan_type: str = "personal", employment_type: str = "salaried"):
    return {"documents": lending.required_documents(loan_type, employment_type)}


@app.get("/lending/glossary", tags=["lending"])
def glossary():
    return lending.GLOSSARY


class CompareRequest(BaseModel):
    offers: list[lending.LoanOffer]


@app.post("/lending/compare", response_model=list[lending.OfferComparison], tags=["lending"])
def compare(body: CompareRequest):
    return lending.compare_loan_offers(body.offers)


class ExplainTextRequest(BaseModel):
    text: str
    language: str = "en"


@app.post("/lending/explain", tags=["lending"])
def explain_loan(body: ExplainTextRequest):
    from ai_engine import get_llm_provider

    return lending.explain_loan_text(body.text, body.language, get_llm_provider())


# ---------------------------------------------------------------------------
# Fintech
# ---------------------------------------------------------------------------
class ValidateRequest(BaseModel):
    field: str
    value: str


@app.post("/fintech/validate", response_model=fintech.ValidationResult, tags=["fintech"])
def validate(body: ValidateRequest):
    return fintech.validate_field(body.field, body.value)


class KYCRequest(BaseModel):
    target_level: str = "full"
    mobile: str = ""
    aadhaar: str = ""
    pan: str = ""
    name_matches_documents: Optional[bool] = None


@app.post("/fintech/kyc/assess", response_model=fintech.KYCAssessment, tags=["fintech"])
def kyc(body: KYCRequest):
    return fintech.assess_kyc(**body.model_dump())


class DisputeRequest(BaseModel):
    description: str
    amount: Optional[float] = None
    transaction_id: str = Field(..., min_length=6, max_length=30, description="UTR / UPI Ref No / Txn ID (required: evidence is checked against it)")
    dispute_type: Optional[str] = None
    transaction_date: Optional[str] = Field(None, description="YYYY-MM-DD")


@app.post("/fintech/disputes", response_model=fintech.DisputeTicket, tags=["fintech"])
def create_dispute(body: DisputeRequest, user_id: str = Depends(current_user_id)):
    """Create a ticket owned by the logged-in user."""
    return fintech.create_dispute_ticket(**body.model_dump(), user_id=user_id)


@app.get("/fintech/disputes/{ticket_id}", response_model=fintech.DisputeTicket, tags=["fintech"])
def get_dispute(ticket_id: str, user_id: str = Depends(current_user_id)):
    return own_ticket(ticket_id, user_id)


class ScamRequest(BaseModel):
    text: str
    sender: Optional[str] = Field(None, description="SMS header (e.g. VM-HDFCBK) or phone number it came from")


@app.post("/fintech/scam-check", response_model=fintech.ScamAssessment, tags=["fintech"])
def scam_check(body: ScamRequest):
    """Checks wording, links, phone numbers and the sender."""
    return fintech.check_scam_message(body.text, body.sender)


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str
    target: str = "hi"


@app.post("/translate", tags=["language"])
def translate(body: TranslateRequest):
    from ai_engine import get_llm_provider

    text, translated = translate_text(body.text, body.target, get_llm_provider())
    return {"text": text, "translated": translated}


# ---------------------------------------------------------------------------
# Accounts (basic details only; no KYC at sign-up)
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    full_name: str
    mobile: str
    password: str
    confirm_password: str
    email: Optional[str] = None
    preferred_language: str = "en"
    accepted_terms: bool = False


class LoginRequest(BaseModel):
    identifier: str = Field(..., description="Mobile number or email")
    password: str


class AuthResponse(BaseModel):
    token: str
    user: accounts.PublicUser


@app.post("/auth/register", response_model=AuthResponse, tags=["auth"])
def register(body: RegisterRequest):
    user = accounts.register_user(**body.model_dump())
    return AuthResponse(token=accounts.issue_token(user.user_id), user=user)


@app.post("/auth/login", response_model=AuthResponse, tags=["auth"])
def login(body: LoginRequest):
    user = accounts.authenticate(body.identifier, body.password)
    return AuthResponse(token=accounts.issue_token(user.user_id), user=user)


@app.get("/me", response_model=accounts.PublicUser, tags=["auth"])
def me(user_id: str = Depends(current_user_id)):
    user = accounts.get_user(user_id)
    if user is None:
        raise HTTPException(404, "User not found.")
    return user


# ---------------------------------------------------------------------------
# Feature upgrades
# ---------------------------------------------------------------------------
@app.get("/policies/{policy_id}/glance", response_model=policy_insights.CoverageGlance, tags=["insurance"])
def glance(policy_id: str):
    return policy_insights.coverage_at_a_glance(get_policy_details(policy_id))


class ExplainTextBody(BaseModel):
    text: str
    language: str = "en"


@app.post("/policies/explain-text", response_model=SimpleExplanation, tags=["insurance"])
def explain_text(body: ExplainTextBody):
    return explain_clause_text(body.text, body.language)


@app.post("/policies/{policy_id}/room-what-if", response_model=list[policy_insights.RoomOption], tags=["insurance"])
def room_what_if(policy_id: str, body: EstimateRequest):
    return policy_insights.room_rent_what_if(get_policy_details(policy_id), body.bill, body.context)


class PrepaymentRequest(EMIRequest):
    extra_monthly: float = 0.0
    yearly_extra: float = 0.0
    lump_sum: float = 0.0
    lump_sum_month: int = 12


@app.post("/lending/prepayment", response_model=lending.PrepaymentPlan, tags=["lending"])
def prepayment(body: PrepaymentRequest):
    return lending.prepayment_plan(body.principal, body.annual_rate_pct, body.tenure_months,
                                   body.extra_monthly, body.yearly_extra, body.lump_sum, body.lump_sum_month)


@app.post("/lending/eligibility/levers", response_model=list[lending.EligibilityLever], tags=["lending"])
def levers(body: lending.LoanApplicant):
    return lending.eligibility_levers(body)


@app.get("/fintech/my-disputes", response_model=list[fintech.DisputeTicket], tags=["fintech"])
def my_disputes(user_id: str = Depends(current_user_id)):
    return fintech.list_dispute_tickets(user_id)


@app.get("/fintech/disputes/{ticket_id}/timeline", response_model=list[fintech.EscalationStep], tags=["fintech"])
def dispute_timeline(ticket_id: str, user_id: str = Depends(current_user_id)):
    return fintech.escalation_timeline(own_ticket(ticket_id, user_id))


@app.get("/fintech/disputes/{ticket_id}/complaint", tags=["fintech"])
def dispute_complaint(ticket_id: str, recipient: str = "app", user_id: str = Depends(current_user_id)):
    user = accounts.get_user(user_id)
    return {"text": fintech.draft_complaint(own_ticket(ticket_id, user_id), user.full_name if user else "", recipient,
                                            user.masked_mobile if user else None)}


# ---------------------------------------------------------------------------
# Evidence attachments (login required; users only reach their own tickets)
# ---------------------------------------------------------------------------
@app.post("/fintech/disputes/{ticket_id}/attachments", response_model=fintech.Attachment, tags=["fintech"])
async def add_ticket_attachment(ticket_id: str, file: UploadFile = File(...), user_id: str = Depends(current_user_id)):
    """Attach a PNG/JPG/PDF screenshot (max 5 MB). The type is verified from the file's content."""
    data = await file.read(fintech.MAX_ATTACHMENT_BYTES + 1)
    user = accounts.get_user(user_id)
    return await run_in_threadpool(fintech.add_attachment, ticket_id, user_id, file.filename or "file", data, None,
                                   user.full_name if user else "")


class SmsBody(BaseModel):
    text: str


@app.post("/fintech/disputes/{ticket_id}/sms", response_model=fintech.Attachment, tags=["fintech"])
def add_ticket_sms(ticket_id: str, body: SmsBody, user_id: str = Depends(current_user_id)):
    """Attach pasted SMS text. OTPs, PINs and card numbers are removed before saving."""
    return fintech.add_sms_text(ticket_id, user_id, body.text)


@app.delete("/fintech/disputes/{ticket_id}/attachments/{attachment_id}", tags=["fintech"])
def remove_ticket_attachment(ticket_id: str, attachment_id: str, user_id: str = Depends(current_user_id)):
    fintech.remove_attachment(ticket_id, user_id, attachment_id)
    return {"removed": attachment_id}


@app.get("/fintech/disputes/{ticket_id}/evidence-pack", tags=["fintech"])
def evidence_pack(ticket_id: str, recipient: str = "app", user_id: str = Depends(current_user_id)):
    """ZIP with the complaint letter, all attachments and a checksum index."""
    from fastapi.responses import Response

    ticket = own_ticket(ticket_id, user_id)
    user = accounts.get_user(user_id)
    data = fintech.build_evidence_pack(ticket, user.full_name if user else "", recipient, user.masked_mobile if user else None)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{ticket_id}_evidence.zip"'})


# ---------------------------------------------------------------------------
# Chat assistant (RAG + MCP tools + guardrails)
# ---------------------------------------------------------------------------
_ASSISTANT = None


def get_assistant():
    global _ASSISTANT
    if _ASSISTANT is None:
        from ai_engine.assistant import Assistant

        _ASSISTANT = Assistant()
    return _ASSISTANT


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)
    language: str = "en"
    policy_id: Optional[str] = Field(None, description="One of the caller's own policies (optional)")


@app.post("/chat", tags=["chat"])
async def chat(body: ChatRequest, user_id: str = Depends(current_user_id)):
    """Saathi assistant. Input and output pass through guardrails; tools run over MCP."""
    from ai_engine.assistant import ChatContext

    user = accounts.get_user(user_id)
    if user is None:
        raise HTTPException(401, "Please log in again.")
    if body.policy_id and body.policy_id not in accounts.user_policy_ids(user_id):
        raise HTTPException(404, "Policy not found.")  # never let a caller query someone else's policy
    ctx = ChatContext(user_id=user_id, user_name=user.first_name, language=body.language, policy_id=body.policy_id,
                      policy_label=body.policy_id or "")
    history = [t.model_dump() for t in body.history if t.role in ("user", "assistant")]
    return await run_in_threadpool(get_assistant().chat, body.message, history, ctx)


@app.get("/languages", tags=["language"])
def languages():
    from ai_engine.language import SUPPORTED_LANGUAGES

    return SUPPORTED_LANGUAGES
