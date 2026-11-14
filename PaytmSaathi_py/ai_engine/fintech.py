"""Fintech journeys: KYC readiness, payment dispute help, scam/phishing check.

Everything is deterministic and runs offline. Identity numbers are validated by
format/checksum only (no external lookup), and are **masked in every output and never
stored or logged**. Regulatory figures are *indicative* and the output says to confirm
them with the provider.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import secrets
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .exceptions import CalculationInputError, EvidenceRejectedError, InvalidFileError, PaytmSaathiError
from .guardrails import redact_sensitive
from .linkcheck import (
    LinkFinding,
    PhoneFinding,
    SenderFinding,
    analyse_link,
    analyse_phones,
    analyse_sender,
    detect_brands,
    extract_urls,
    load_blocklist,
)
from .utils import POLICIES_DIR, ensure_dir

DISPUTES_DIR = str(Path(POLICIES_DIR).parent / "disputes")


# ---------------------------------------------------------------------------
# Identity / payment-identifier validators (format only)
# ---------------------------------------------------------------------------
_PAN_TYPES = {"P": "Individual", "C": "Company", "H": "Hindu Undivided Family", "F": "Firm",
              "A": "Association of Persons", "T": "Trust", "B": "Body of Individuals",
              "L": "Local Authority", "J": "Artificial Juridical Person", "G": "Government"}

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff_valid(number: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def mask_id(value: str, keep: int = 4) -> str:
    value = value or ""
    return "X" * max(len(value) - keep, 0) + value[-keep:]


class ValidationResult(BaseModel):
    field: str
    valid: bool
    masked_value: str
    message: str
    message_hi: str
    detail: Optional[str] = None


def validate_field(field: str, value: str) -> ValidationResult:
    """Validate one identifier: pan | aadhaar | ifsc | upi | mobile | account."""
    raw = (value or "").strip()
    field = field.lower().strip()
    if field == "pan":
        v = raw.upper()
        ok = bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", v))
        detail = _PAN_TYPES.get(v[3]) if ok and len(v) > 3 else None
        return ValidationResult(field=field, valid=ok, masked_value=mask_id(v), detail=detail,
                                message="PAN format looks correct." if ok else "PAN must look like ABCDE1234F (5 letters, 4 digits, 1 letter).",
                                message_hi="पैन का फॉर्मेट सही है।" if ok else "पैन ABCDE1234F जैसा होना चाहिए (5 अक्षर, 4 अंक, 1 अक्षर)।")
    if field == "aadhaar":
        v = re.sub(r"[\s-]", "", raw)
        ok = bool(re.fullmatch(r"[2-9][0-9]{11}", v)) and _verhoeff_valid(v)
        return ValidationResult(field=field, valid=ok, masked_value=mask_id(v),
                                message="Aadhaar number passes the format and checksum test." if ok else "Aadhaar must be 12 digits (not starting with 0 or 1) with a valid checksum.",
                                message_hi="आधार नंबर का फॉर्मेट और चेकसम सही है।" if ok else "आधार 12 अंकों का होना चाहिए (0 या 1 से शुरू नहीं) और चेकसम सही होना चाहिए।")
    if field == "ifsc":
        v = raw.upper()
        ok = bool(re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", v))
        return ValidationResult(field=field, valid=ok, masked_value=v,
                                message="IFSC format looks correct." if ok else "IFSC must be 11 characters: 4 letters, a zero, then 6 letters/digits.",
                                message_hi="आईएफएससी का फॉर्मेट सही है।" if ok else "आईएफएससी 11 अक्षरों का होता है: 4 अक्षर, एक शून्य, फिर 6 अक्षर/अंक।")
    if field == "upi":
        ok = bool(re.fullmatch(r"[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}", raw))
        return ValidationResult(field=field, valid=ok, masked_value=raw,
                                message="UPI ID format looks correct. Always verify the payee name shown before paying." if ok else "UPI ID must look like name@bank.",
                                message_hi="यूपीआई आईडी का फॉर्मेट सही है। भुगतान से पहले दिखाया गया नाम ज़रूर जाँचें।" if ok else "यूपीआई आईडी name@bank जैसी होनी चाहिए।")
    if field == "mobile":
        v = re.sub(r"[\s-]", "", raw)
        v = re.sub(r"^(\+91|91|0)(?=\d{10}$)", "", v)
        ok = bool(re.fullmatch(r"[6-9][0-9]{9}", v))
        return ValidationResult(field=field, valid=ok, masked_value=mask_id(v, 2),
                                message="Mobile number looks valid." if ok else "Indian mobile numbers have 10 digits and start with 6-9.",
                                message_hi="मोबाइल नंबर सही लगता है।" if ok else "भारतीय मोबाइल नंबर 10 अंकों के होते हैं और 6-9 से शुरू होते हैं।")
    if field == "account":
        v = re.sub(r"[\s-]", "", raw)
        ok = bool(re.fullmatch(r"[0-9]{9,18}", v))
        return ValidationResult(field=field, valid=ok, masked_value=mask_id(v),
                                message="Account number length looks valid." if ok else "Bank account numbers have 9 to 18 digits.",
                                message_hi="खाता संख्या की लंबाई सही है।" if ok else "बैंक खाता संख्या 9 से 18 अंकों की होती है।")
    raise CalculationInputError("Unknown field. Use one of: pan, aadhaar, ifsc, upi, mobile, account.")


# ---------------------------------------------------------------------------
# KYC readiness
# ---------------------------------------------------------------------------
KYC_LEVELS = {
    "minimum": {
        "title": "Minimum KYC", "title_hi": "न्यूनतम केवाईसी",
        "needs": ["mobile", "aadhaar_or_other_id"],
        "limits": "Low balance and monthly limits; usually cannot withdraw to bank or transfer freely.",
        "limits_hi": "कम बैलेंस और मासिक सीमा; आमतौर पर बैंक में निकासी/मुक्त ट्रांसफर नहीं।",
    },
    "full": {
        "title": "Full KYC", "title_hi": "पूर्ण केवाईसी",
        "needs": ["mobile", "aadhaar", "pan", "video_or_in_person_verification"],
        "limits": "Higher balance and transfer limits and access to more financial products.",
        "limits_hi": "ज़्यादा बैलेंस/ट्रांसफर सीमा और ज़्यादा वित्तीय सेवाओं तक पहुँच।",
    },
}
KYC_NOTE = "Limits are set by RBI and your provider and change over time; confirm current limits in the app."
KYC_NOTE_HI = "सीमाएँ आरबीआई और आपके प्रदाता तय करते हैं और बदलती रहती हैं; ताज़ा सीमा ऐप में देखें।"


class KYCAssessment(BaseModel):
    target_level: str
    ready: bool
    checks: list[ValidationResult] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    missing_hi: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    next_steps_hi: list[str] = Field(default_factory=list)
    level_info: dict = Field(default_factory=dict)
    note: str = KYC_NOTE
    note_hi: str = KYC_NOTE_HI


def assess_kyc(target_level: str, mobile: str = "", aadhaar: str = "", pan: str = "",
               name_matches_documents: Optional[bool] = None) -> KYCAssessment:
    """Check which KYC documents are valid/missing. Identifiers are masked in the output."""
    if target_level not in KYC_LEVELS:
        raise CalculationInputError("target_level must be 'minimum' or 'full'.")
    out = KYCAssessment(target_level=target_level, ready=False, level_info=KYC_LEVELS[target_level])
    labels = {"mobile": ("Mobile number", "मोबाइल नंबर"), "aadhaar": ("Aadhaar", "आधार"), "pan": ("PAN", "पैन")}
    values = {"mobile": mobile, "aadhaar": aadhaar, "pan": pan}
    required = ["mobile", "aadhaar"] + (["pan"] if target_level == "full" else [])
    for key in required:
        if not values[key].strip():
            out.missing.append(f"{labels[key][0]} is required")
            out.missing_hi.append(f"{labels[key][1]} ज़रूरी है")
            continue
        result = validate_field(key, values[key])
        out.checks.append(result)
        if not result.valid:
            out.missing.append(f"{labels[key][0]}: {result.message}")
            out.missing_hi.append(f"{labels[key][1]}: {result.message_hi}")
    if name_matches_documents is False:
        out.missing.append("Name must match exactly across Aadhaar and PAN")
        out.missing_hi.append("आधार और पैन में नाम बिल्कुल एक जैसा होना चाहिए")
    out.ready = not out.missing
    if out.ready:
        out.next_steps = ["Keep your original documents and a working phone for OTP.",
                          "Complete OTP-based Aadhaar verification in the app."]
        out.next_steps_hi = ["अपने मूल दस्तावेज़ और ओटीपी के लिए चालू फोन तैयार रखें।",
                             "ऐप में ओटीपी से आधार सत्यापन पूरा करें।"]
        if target_level == "full":
            out.next_steps.append("Finish the video or in-person verification step.")
            out.next_steps_hi.append("वीडियो या सामने से सत्यापन का चरण पूरा करें।")
    else:
        out.next_steps = ["Correct the items above, then check again.",
                          "Never share your OTP, PIN or full document numbers with anyone claiming to help with KYC."]
        out.next_steps_hi = ["ऊपर दी गई गलतियाँ ठीक करके दोबारा जाँचें।",
                             "केवाईसी में मदद का दावा करने वाले किसी को भी अपना ओटीपी, पिन या पूरे दस्तावेज़ नंबर न बताएँ।"]
    return out


# ---------------------------------------------------------------------------
# Payment disputes
# ---------------------------------------------------------------------------
DISPUTE_TYPES: dict[str, dict] = {
    "money_debited_failed": {
        "title": "Money debited but payment failed", "title_hi": "पैसा कटा पर पेमेंट फेल",
        "keywords": ["debited", "deducted", "failed", "pending", "not received by", "money cut", "कट गया", "कटा", "फेल", "पेंडिंग"],
        "timeline": "Failed transactions are generally auto-reversed within 1-5 working days under RBI's turnaround-time rules; delayed reversals may earn compensation.",
        "timeline_hi": "आरबीआई के नियमों के तहत फेल लेन-देन आमतौर पर 1-5 कार्य दिवस में अपने-आप वापस हो जाते हैं; देरी पर मुआवज़ा मिल सकता है।",
        "steps": ["Wait 24 hours - most failed payments reverse automatically.",
                  "Note the transaction ID / UTR, amount, date and time.",
                  "Raise a complaint in the app under the transaction (Help / Report issue).",
                  "If not resolved in 5 working days, write to your bank's nodal officer, then the RBI Ombudsman (cms.rbi.org.in)."],
        "steps_hi": ["24 घंटे इंतज़ार करें - ज़्यादातर फेल पेमेंट अपने-आप वापस आ जाते हैं।",
                     "लेन-देन आईडी / यूटीआर, राशि, तारीख और समय नोट करें।",
                     "ऐप में उस लेन-देन के नीचे शिकायत दर्ज करें (Help / Report issue)।",
                     "5 कार्य दिवस में हल न हो तो बैंक के नोडल अधिकारी, फिर आरबीआई लोकपाल (cms.rbi.org.in) को लिखें।"],
        "evidence": ["Transaction ID / UTR", "Screenshot of debit SMS or statement", "Amount, date and time"],
        "urgent": False,
    },
    "wrong_recipient": {
        "title": "Money sent to the wrong person", "title_hi": "गलत व्यक्ति को पैसा चला गया",
        "keywords": ["wrong number", "wrong person", "wrong account", "wrong upi", "mistake", "galat", "गलत", "ग़लत"],
        "timeline": "Recovery depends on the recipient agreeing to return the money; banks can help coordinate but cannot always reverse it.",
        "timeline_hi": "पैसा वापस मिलना पाने वाले की सहमति पर निर्भर करता है; बैंक मदद कर सकता है पर हमेशा रिवर्स नहीं कर सकता।",
        "steps": ["Report it in the app immediately with the transaction ID.",
                  "Contact your bank's customer care and request a recovery/complaint ticket.",
                  "Do not send further money to 'release' the amount - that is a common scam.",
                  "If unresolved, escalate to the bank's nodal officer and the RBI Ombudsman."],
        "steps_hi": ["तुरंत लेन-देन आईडी के साथ ऐप में रिपोर्ट करें।",
                     "बैंक कस्टमर केयर से संपर्क कर रिकवरी/शिकायत टिकट बनवाएँ।",
                     "पैसा 'छुड़वाने' के नाम पर और पैसा न भेजें - यह आम ठगी है।",
                     "हल न हो तो बैंक के नोडल अधिकारी और आरबीआई लोकपाल तक जाएँ।"],
        "evidence": ["Transaction ID / UTR", "Recipient UPI ID or number", "Screenshot of the payment"],
        "urgent": False,
    },
    "duplicate_charge": {
        "title": "Charged twice for one payment", "title_hi": "एक पेमेंट पर दो बार पैसा कटा",
        "keywords": ["twice", "double", "duplicate", "two times", "charged again", "दो बार", "डबल"],
        "timeline": "Merchants usually refund duplicate charges in 5-7 working days.",
        "timeline_hi": "मर्चेंट आमतौर पर दोहरे चार्ज को 5-7 कार्य दिवस में रिफंड करते हैं।",
        "steps": ["Check whether the second charge is only a pending authorisation (it may drop off).",
                  "Contact the merchant with both transaction IDs and ask for a refund.",
                  "Raise a dispute in the app if the merchant does not respond in 3 days."],
        "steps_hi": ["देखें कि दूसरा चार्ज सिर्फ पेंडिंग तो नहीं (वह अपने-आप हट सकता है)।",
                     "दोनों लेन-देन आईडी के साथ मर्चेंट से रिफंड माँगें।",
                     "मर्चेंट 3 दिन में जवाब न दे तो ऐप में शिकायत दर्ज करें।"],
        "evidence": ["Both transaction IDs", "Merchant name and invoice", "Statement showing both debits"],
        "urgent": False,
    },
    "refund_not_received": {
        "title": "Refund not received", "title_hi": "रिफंड नहीं मिला",
        "keywords": ["refund", "return", "cancelled order", "not credited", "रिफंड", "वापस नहीं"],
        "timeline": "Refunds typically reach your account in 5-10 working days after the merchant initiates them.",
        "timeline_hi": "मर्चेंट के शुरू करने के बाद रिफंड आमतौर पर 5-10 कार्य दिवस में खाते में आता है।",
        "steps": ["Ask the merchant for the refund reference number and date initiated.",
                  "Check your original payment source (bank / wallet / card).",
                  "If past 10 working days, raise a dispute in the app with the refund reference."],
        "steps_hi": ["मर्चेंट से रिफंड रेफरेंस नंबर और शुरू होने की तारीख पूछें।",
                     "अपना मूल भुगतान स्रोत (बैंक / वॉलेट / कार्ड) देखें।",
                     "10 कार्य दिवस के बाद रिफंड रेफरेंस के साथ ऐप में शिकायत दर्ज करें।"],
        "evidence": ["Order / invoice number", "Refund reference from merchant", "Original payment ID"],
        "urgent": False,
    },
    "unauthorized_fraud": {
        "title": "Unauthorised transaction or fraud", "title_hi": "बिना अनुमति का लेन-देन या धोखाधड़ी",
        "keywords": ["fraud", "scam", "unauthorized", "unauthorised", "not me", "hacked", "stolen", "otp shared",
                     "cheated", "phishing", "धोखा", "ठगी", "फ्रॉड", "हैक", "मैंने नहीं किया"],
        "timeline": "Reporting within 3 days of the fraud gives you the strongest protection under RBI's customer-liability rules.",
        "timeline_hi": "धोखाधड़ी के 3 दिन के भीतर रिपोर्ट करने पर आरबीआई के नियमों में आपको सबसे ज़्यादा सुरक्षा मिलती है।",
        "steps": ["Call the national cyber-fraud helpline 1930 right away (faster reporting helps freeze the money).",
                  "Block the wallet/UPI/card in the app or by calling customer care; change your UPI PIN and passwords.",
                  "File a complaint at cybercrime.gov.in and keep the acknowledgement number.",
                  "Inform your bank in writing and ask for the unauthorised-transaction investigation.",
                  "Do not share OTP, PIN or install any screen-sharing app, even if the caller says they are from support."],
        "steps_hi": ["तुरंत राष्ट्रीय साइबर-फ्रॉड हेल्पलाइन 1930 पर कॉल करें (जल्दी रिपोर्ट से पैसा फ्रीज़ हो सकता है)।",
                     "ऐप या कस्टमर केयर से वॉलेट/यूपीआई/कार्ड ब्लॉक करें; यूपीआई पिन और पासवर्ड बदलें।",
                     "cybercrime.gov.in पर शिकायत दर्ज करें और पावती नंबर सँभालकर रखें।",
                     "बैंक को लिखित सूचना दें और अनधिकृत लेन-देन की जाँच माँगें।",
                     "ओटीपी/पिन किसी को न बताएँ और स्क्रीन-शेयरिंग ऐप इंस्टॉल न करें, चाहे कॉलर खुद को सपोर्ट बताए।"],
        "evidence": ["Transaction IDs of all fraudulent debits", "Screenshots of messages/calls", "Caller number or UPI ID of the fraudster"],
        "urgent": True,
    },
    "other": {
        "title": "Other payment issue", "title_hi": "अन्य पेमेंट समस्या",
        "keywords": [],
        "timeline": "Most complaints get a first response within a few working days.",
        "timeline_hi": "ज़्यादातर शिकायतों का पहला जवाब कुछ कार्य दिवसों में मिल जाता है।",
        "steps": ["Collect the transaction ID, date and amount.", "Raise a complaint in the app.",
                  "If unresolved in 30 days, escalate to the bank's nodal officer and then the RBI Ombudsman (cms.rbi.org.in)."],
        "steps_hi": ["लेन-देन आईडी, तारीख और राशि जमा करें।", "ऐप में शिकायत दर्ज करें।",
                     "30 दिन में हल न हो तो बैंक के नोडल अधिकारी और फिर आरबीआई लोकपाल (cms.rbi.org.in) तक जाएँ।"],
        "evidence": ["Transaction ID", "Description of the issue"],
        "urgent": False,
    },
}


class DisputeGuidance(BaseModel):
    dispute_type: str
    title: str
    title_hi: str
    confidence: str  # high | medium | low
    urgent: bool
    steps: list[str]
    steps_hi: list[str]
    evidence_to_collect: list[str]
    expected_timeline: str
    expected_timeline_hi: str
    helplines: list[str] = Field(default_factory=lambda: ["Cyber fraud helpline: 1930", "cybercrime.gov.in", "RBI Ombudsman: cms.rbi.org.in"])


def classify_dispute(description: str) -> tuple[str, str]:
    """Keyword-based classification. Returns (dispute_type, confidence). Fraud wins ties."""
    text = (description or "").lower()
    if not text.strip():
        raise CalculationInputError("Describe the problem to get help.")
    scores = {k: sum(1 for kw in v["keywords"] if kw.lower() in text) for k, v in DISPUTE_TYPES.items()}
    best = max(scores, key=lambda k: (scores[k], k == "unauthorized_fraud"))
    if scores[best] == 0:
        return "other", "low"
    return best, "high" if scores[best] >= 2 else "medium"


def dispute_guidance(dispute_type: str, confidence: str = "high") -> DisputeGuidance:
    if dispute_type not in DISPUTE_TYPES:
        raise CalculationInputError(f"Unknown dispute type. Use one of: {', '.join(DISPUTE_TYPES)}.")
    d = DISPUTE_TYPES[dispute_type]
    return DisputeGuidance(
        dispute_type=dispute_type, title=d["title"], title_hi=d["title_hi"], confidence=confidence,
        urgent=d["urgent"], steps=d["steps"], steps_hi=d["steps_hi"], evidence_to_collect=d["evidence"],
        expected_timeline=d["timeline"], expected_timeline_hi=d["timeline_hi"])


class Attachment(BaseModel):
    """Evidence attached to a ticket: a screenshot/PDF, or pasted SMS text."""

    attachment_id: str
    kind: str  # screenshot | document | sms
    filename: str
    size: int = 0
    sha256: str = ""
    added_at: str
    text: Optional[str] = None  # OCR text or pasted SMS, with credentials removed
    hints: dict = Field(default_factory=dict)  # utr / amount / date found in the text
    verified: bool = False  # passed the transaction-ID / name / date checks
    checks: list[str] = Field(default_factory=list)  # which checks passed


class DisputeTicket(BaseModel):
    ticket_id: str
    dispute_type: str
    description: str
    amount: Optional[float] = None
    transaction_id: Optional[str] = None
    status: str = "open"
    created_at: str
    transaction_date: Optional[str] = None  # YYYY-MM-DD, when the payment happened
    user_id: Optional[str] = None
    attachments: list[Attachment] = Field(default_factory=list)
    guidance: DisputeGuidance


def create_dispute_ticket(description: str, amount: Optional[float] = None,
                          transaction_id: Optional[str] = None,
                          dispute_type: Optional[str] = None,
                          directory: Optional[str] = None,
                          user_id: Optional[str] = None,
                          transaction_date: Optional[str] = None) -> DisputeTicket:
    """Classify (unless a type is given), build guidance and save a local ticket record."""
    if amount is not None and amount < 0:
        raise CalculationInputError("Amount cannot be negative.")
    if transaction_id:
        transaction_id = validate_transaction_id(transaction_id)
    if dispute_type:
        kind, confidence = dispute_type, "high"
    else:
        kind, confidence = classify_dispute(description)
    now = datetime.now(timezone.utc)
    ticket = DisputeTicket(
        ticket_id=f"DSP-{now:%Y%m%d}-{secrets.token_hex(3).upper()}", dispute_type=kind,
        description=(description or "")[:2000], amount=amount,
        transaction_id=(transaction_id or None), created_at=now.isoformat(timespec="seconds"),
        transaction_date=transaction_date, user_id=user_id,
        guidance=dispute_guidance(kind, confidence))
    folder = ensure_dir(directory or DISPUTES_DIR)
    (folder / f"{ticket.ticket_id}.json").write_text(ticket.model_dump_json(indent=2), encoding="utf-8")
    return ticket


def list_dispute_tickets(user_id: str, directory: Optional[str] = None) -> list[DisputeTicket]:
    """The user's own tickets, newest first."""
    folder = Path(directory or DISPUTES_DIR)
    if not folder.exists():
        return []
    tickets = []
    for path in folder.glob("DSP-*.json"):
        try:
            ticket = DisputeTicket.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        if ticket.user_id == user_id:
            tickets.append(ticket)
    return sorted(tickets, key=lambda t: t.created_at, reverse=True)


# Escalation ladder per dispute type: (days after reporting, what happens / what to do).
# Timelines follow RBI's TAT and Integrated Ombudsman rules; they are indicative.
_ESCALATION: dict[str, list[tuple[int, str, str]]] = {
    "money_debited_failed": [
        (0, "You reported the issue", "आपने समस्या दर्ज की"),
        (1, "UPI reversal is due by now (T+1)", "यूपीआई रिवर्सल अब तक हो जाना चाहिए (T+1)"),
        (5, "Not reversed? Escalate to the bank's nodal officer; compensation may apply", "वापस नहीं आया? बैंक के नोडल अधिकारी को लिखें; मुआवज़ा मिल सकता है"),
        (30, "No reply in 30 days? Complain to the RBI Ombudsman (cms.rbi.org.in)", "30 दिन में जवाब नहीं? आरबीआई लोकपाल (cms.rbi.org.in) में शिकायत करें"),
    ],
    "wrong_recipient": [
        (0, "You reported the issue with the transaction ID", "आपने लेन-देन आईडी के साथ शिकायत दर्ज की"),
        (7, "Follow up with your bank on the recovery request", "रिकवरी अनुरोध पर बैंक से फॉलो-अप करें"),
        (30, "No resolution? Complain to the RBI Ombudsman", "हल नहीं? आरबीआई लोकपाल में शिकायत करें"),
    ],
    "duplicate_charge": [
        (0, "You contacted the merchant", "आपने मर्चेंट से संपर्क किया"),
        (3, "No reply from merchant? Raise a dispute in the app", "मर्चेंट का जवाब नहीं? ऐप में शिकायत दर्ज करें"),
        (7, "Refund should normally be credited by now", "रिफंड आमतौर पर अब तक आ जाना चाहिए"),
        (30, "Still not refunded? Complain to the RBI Ombudsman", "अब भी रिफंड नहीं? आरबीआई लोकपाल में शिकायत करें"),
    ],
    "refund_not_received": [
        (0, "You asked the merchant for the refund reference", "आपने मर्चेंट से रिफंड रेफरेंस माँगा"),
        (10, "Refund should be credited by now; raise a dispute if not", "रिफंड अब तक आ जाना चाहिए; न आए तो शिकायत करें"),
        (30, "No resolution? Complain to the RBI Ombudsman", "हल नहीं? आरबीआई लोकपाल में शिकायत करें"),
    ],
    "unauthorized_fraud": [
        (0, "Call 1930, block your accounts and report to your bank today", "आज ही 1930 पर कॉल करें, खाते ब्लॉक करें और बैंक को बताएँ"),
        (3, "Reporting within 3 days gives you the strongest (zero-liability) protection", "3 दिन में रिपोर्ट करने पर सबसे मज़बूत (शून्य देयता) सुरक्षा मिलती है"),
        (10, "Bank should credit the disputed amount (shadow reversal) within 10 working days", "बैंक को 10 कार्य दिवस में विवादित राशि (शैडो रिवर्सल) जमा करनी चाहिए"),
        (90, "Bank must resolve the complaint within 90 days", "बैंक को 90 दिन में शिकायत हल करनी होगी"),
    ],
    "other": [
        (0, "You raised the complaint", "आपने शिकायत दर्ज की"),
        (30, "No resolution? Complain to the RBI Ombudsman", "हल नहीं? आरबीआई लोकपाल में शिकायत करें"),
    ],
}


class EscalationStep(BaseModel):
    day: int
    due_on: str
    text: str
    text_hi: str
    reached: bool


def escalation_timeline(ticket: DisputeTicket, today: Optional[datetime] = None) -> list[EscalationStep]:
    """Concrete dates for each escalation step, counted from the ticket date."""
    from datetime import timedelta

    start = datetime.fromisoformat(ticket.created_at).date()
    today_d = (today or datetime.now(timezone.utc)).date()
    return [EscalationStep(day=day, due_on=(start + timedelta(days=day)).isoformat(), text=en, text_hi=hi,
                           reached=(start + timedelta(days=day)) <= today_d)
            for day, en, hi in _ESCALATION.get(ticket.dispute_type, _ESCALATION["other"])]


_RECIPIENTS = {
    "app": ("Customer Support", "app support"),
    "bank": ("The Nodal Officer", "bank's nodal officer"),
    "ombudsman": ("The RBI Ombudsman (via cms.rbi.org.in)", "RBI Ombudsman"),
}


def draft_complaint(ticket: DisputeTicket, customer_name: str, recipient: str = "app",
                    masked_mobile: Optional[str] = None) -> str:
    """Ready-to-send complaint text prefilled from the ticket (no sensitive numbers)."""
    if recipient not in _RECIPIENTS:
        raise CalculationInputError("recipient must be 'app', 'bank' or 'ombudsman'.")
    to_line, _ = _RECIPIENTS[recipient]
    amount = f"Rs. {ticket.amount:,.2f}" if ticket.amount else "[amount]"
    txn = ticket.transaction_id or "[transaction ID / UTR]"
    when = ticket.transaction_date or "[date of transaction]"
    lines = [
        f"To: {to_line}",
        f"Subject: {ticket.guidance.title} - {amount} - Ref {ticket.ticket_id}",
        "",
        "Dear Sir/Madam,",
        "",
        f"I am writing about a payment of {amount} made on {when} (transaction ID {txn}).",
        f"Issue: {ticket.guidance.title}.",
        f"Details: {ticket.description.strip()}",
        "",
    ]
    if recipient == "app":
        lines.append("Please investigate and reverse/refund the amount at the earliest.")
    elif recipient == "bank":
        lines.append(f"I first raised this on {ticket.created_at[:10]} (reference {ticket.ticket_id}) and it is "
                     "still unresolved. As per RBI's turnaround-time framework, I request resolution and any "
                     "compensation due for the delay.")
    else:
        lines.append(f"I complained to my bank/payment provider on {ticket.created_at[:10]} (reference "
                     f"{ticket.ticket_id}) and have not received a satisfactory resolution within 30 days. "
                     "I request your intervention under the RBI Integrated Ombudsman Scheme.")
    if ticket.dispute_type == "unauthorized_fraud":
        lines.append("I did not authorise this transaction and have reported it to the cyber-crime helpline 1930.")
    lines += [""] + _attachment_lines(ticket) + ["", "Regards,", customer_name or "[your name]"]
    if masked_mobile:
        lines.append(f"Registered mobile: {masked_mobile}")
    return "\n".join(lines)


def get_dispute_ticket(ticket_id: str, directory: Optional[str] = None) -> DisputeTicket:
    if not re.fullmatch(r"DSP-\d{8}-[A-F0-9]{6}", ticket_id or ""):
        raise CalculationInputError("Invalid ticket ID.")
    path = Path(directory or DISPUTES_DIR) / f"{ticket_id}.json"
    if not path.exists():
        raise PaytmSaathiError(f"Ticket not found: {ticket_id}", "ticket_not_found")
    return DisputeTicket.model_validate(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Evidence: screenshots, documents and pasted SMS attached to a ticket
# ---------------------------------------------------------------------------
MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
MAX_ATTACHMENTS = 6
_MAGIC = {".png": (b"\x89PNG\r\n\x1a\n",), ".jpg": (b"\xff\xd8\xff",), ".jpeg": (b"\xff\xd8\xff",), ".pdf": (b"%PDF-",)}
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _owned_ticket(ticket_id: str, user_id: str, directory: Optional[str]) -> DisputeTicket:
    ticket = get_dispute_ticket(ticket_id, directory)
    if not user_id or ticket.user_id != user_id:
        raise PaytmSaathiError("Ticket not found: " + ticket_id, "ticket_not_found")  # don't reveal other users' tickets
    return ticket


def _save_ticket(ticket: DisputeTicket, directory: Optional[str]) -> None:
    folder = ensure_dir(directory or DISPUTES_DIR)
    (folder / f"{ticket.ticket_id}.json").write_text(ticket.model_dump_json(indent=2), encoding="utf-8")


def _attachment_dir(ticket_id: str, directory: Optional[str]) -> Path:
    return ensure_dir(Path(directory or DISPUTES_DIR) / "attachments" / ticket_id)


def extract_evidence_hints(text: str) -> dict:
    """Pull the transaction reference, amount and date out of SMS / screenshot text."""
    hints: dict = {}
    utr = re.search(r"(?:utr|rrn|ref(?:erence)?(?:\s*no)?|txn(?:\s*id)?|transaction\s*(?:id|no))\W{0,6}(\d{9,16})", text, re.IGNORECASE)
    if utr:
        hints["utr"] = utr.group(1)
    amount = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)", text, re.IGNORECASE)
    if amount:
        try:
            hints["amount"] = float(amount.group(1).replace(",", ""))
        except ValueError:
            pass
    # No \b: OCR often glues words together ("on01/11/2026to"), so only require non-digits around it.
    d = re.search(r"(?<!\d)(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4}|\d{2})(?!\d)", text)
    if d:
        day, month, year = int(d.group(1)), int(d.group(2)), int(d.group(3))
        year += 2000 if year < 100 else 0
        try:
            hints["date"] = datetime(year, month, day).date().isoformat()
        except ValueError:
            pass
    else:
        d = re.search(r"\b(\d{1,2})[\s-]([A-Za-z]{3})[a-z]*[\s,-]+(\d{4})\b", text)
        if d and d.group(2).lower() in _MONTHS:
            try:
                hints["date"] = datetime(int(d.group(3)), _MONTHS[d.group(2).lower()], int(d.group(1))).date().isoformat()
            except ValueError:
                pass
    return hints


def _safe_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "file").name)[:80].strip("._") or "file"
    return base


# ---------------------------------------------------------------------------
# Evidence verification: does the SMS / screenshot belong to THIS ticket and THIS user?
# ---------------------------------------------------------------------------
_TXN_FORMAT = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{4,28}[A-Za-z0-9]")
# Letters OCR commonly reads instead of digits (only applied when the ID is all digits).
_OCR_DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})
_TXN_LABELLED = re.compile(
    r"(?:utr|rrn|upi\s*ref(?:erence)?|ref(?:erence)?|txn|transaction)\s*(?:no\.?|number|id|#)?\s*[:.\-]?\s*([A-Za-z0-9]{6,30})",
    re.IGNORECASE)
_DATE_PATTERNS = [
    r"(?<!\d)\d{1,2}[/\-.]\d{1,2}[/\-.](?:\d{4}|\d{2})(?!\d)",                    # 01/11/2026, 01-11-26
    r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)",                                              # 2026-11-01
    r"(?<!\d)\d{1,2}[\s\-]?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s\-,]*(?:\d{4}|\d{2})(?!\d)",  # 01 Nov 2026, 01-Nov-26
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}",  # Nov 1, 2026
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS), re.IGNORECASE)
_TIME_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm|hrs?)?", re.IGNORECASE)


def normalise_txn_id(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def validate_transaction_id(value: str) -> str:
    """Return the cleaned ID or raise. UTRs are 12 digits; wallets/apps use letters too."""
    value = (value or "").strip()
    if not _TXN_FORMAT.fullmatch(value):
        raise CalculationInputError("Enter the transaction ID / UTR exactly as shown (6-30 letters or digits).")
    return value


def txn_id_in_text(txn_id: str, text: str) -> bool:
    """True if the ticket's transaction ID appears in the text (spaces/punctuation and OCR slips ignored)."""
    want, hay = normalise_txn_id(txn_id), normalise_txn_id(text)
    if len(want) < 6 or not hay:
        return False
    if want in hay:
        return True
    return want.isdigit() and want in hay.translate(_OCR_DIGIT_FIX)


def find_txn_ids(text: str) -> list[str]:
    """Transaction IDs that are labelled as such (UTR / Ref No / Txn ID ...) in the text."""
    return [m.group(1) for m in _TXN_LABELLED.finditer(text or "") if re.search(r"\d", m.group(1))]


def has_timestamp(text: str) -> bool:
    return bool(_DATE_RE.search(text or ""))


def name_in_text(full_name: str, text: str) -> tuple[bool, list[str]]:
    """Every part of the user's name (3+ letters) must appear; one OCR slip allowed in long parts."""
    from .linkcheck import _edit_distance

    parts = [normalise_txn_id(p) for p in (full_name or "").split()]
    parts = [p for p in parts if len(p) >= 3] or [p for p in parts if p]
    hay = normalise_txn_id(text)
    missing = []
    for part in parts:
        if part in hay:
            continue
        n = len(part)
        if n >= 5 and any(_edit_distance(hay[i:i + n], part, cap=1) <= 1 for i in range(max(len(hay) - n + 1, 0))):
            continue
        missing.append(part.title())
    return (bool(parts) and not missing), missing


def _txn_reasons(ticket: DisputeTicket, text: str, what: str, what_hi: str) -> tuple[list[str], list[str], list[str]]:
    """(reasons_en, reasons_hi, passed_checks) for the transaction-ID check."""
    if txn_id_in_text(ticket.transaction_id or "", text):
        return [], [], [f"Transaction ID {ticket.transaction_id} matches"]
    found = [f for f in find_txn_ids(text) if normalise_txn_id(f) != normalise_txn_id(ticket.transaction_id or "")]
    if found:
        return ([f"The transaction ID in this {what} ({found[0]}) does not match your ticket ({ticket.transaction_id})."],
                [f"इस {what_hi} में लेन-देन आईडी ({found[0]}) आपके टिकट ({ticket.transaction_id}) से मेल नहीं खाती।"], [])
    return ([f"This {what} does not show your transaction ID {ticket.transaction_id}."],
            [f"इस {what_hi} में आपकी लेन-देन आईडी {ticket.transaction_id} नहीं दिखती।"], [])


def verify_sms(ticket: DisputeTicket, text: str) -> list[str]:
    """SMS must contain a date/time and the ticket's transaction ID. Raises EvidenceRejectedError; returns passed checks."""
    reasons, reasons_hi, passed = [], [], []
    if has_timestamp(text):
        passed.append("Date/time found" if _TIME_RE.search(text) else "Date found")
    else:
        reasons.append("The SMS must include the transaction date (and time, if shown).")
        reasons_hi.append("एसएमएस में लेन-देन की तारीख (और समय, यदि हो) होनी चाहिए।")
    r, rh, p = _txn_reasons(ticket, text, "SMS", "एसएमएस")
    reasons, reasons_hi, passed = reasons + r, reasons_hi + rh, passed + p
    if reasons:
        raise EvidenceRejectedError(reasons, reasons_hi)
    return passed


def verify_document(ticket: DisputeTicket, text: Optional[str], full_name: str, kind: str) -> list[str]:
    """Screenshot/PDF must show the ticket's transaction ID and the user's name."""
    what, what_hi = ("screenshot", "स्क्रीनशॉट") if kind == "screenshot" else ("document", "दस्तावेज़")
    if not text or not text.strip():
        raise EvidenceRejectedError(
            [f"We could not read any text in this {what}, so we can't confirm it is yours. Upload a clearer image."],
            [f"इस {what_hi} का कोई टेक्स्ट पढ़ा नहीं जा सका, इसलिए पुष्टि नहीं हो सकती। साफ़ तस्वीर अपलोड करें।"])
    reasons, reasons_hi, passed = _txn_reasons(ticket, text, what, what_hi)
    ok, missing = name_in_text(full_name, text)
    if ok:
        passed.append("Your name matches")
    else:
        reasons.append(f"This {what} does not show your name ({full_name}); missing: {', '.join(missing) or full_name}.")
        reasons_hi.append(f"इस {what_hi} में आपका नाम ({full_name}) नहीं दिखता।")
    if reasons:
        raise EvidenceRejectedError(reasons, reasons_hi)
    return passed


def _require_txn(ticket: DisputeTicket) -> None:
    if not ticket.transaction_id:
        raise EvidenceRejectedError(
            ["Add the transaction ID to this ticket first - evidence is checked against it."],
            ["पहले इस टिकट में लेन-देन आईडी जोड़ें - सबूत की जाँच उसी से होती है।"])


def set_transaction_id(ticket_id: str, user_id: str, transaction_id: str, directory: Optional[str] = None) -> DisputeTicket:
    """Add the transaction ID to an older ticket. Not allowed once evidence was checked against another ID."""
    ticket = _owned_ticket(ticket_id, user_id, directory)
    value = validate_transaction_id(transaction_id)
    if ticket.attachments and normalise_txn_id(value) != normalise_txn_id(ticket.transaction_id or ""):
        raise CalculationInputError("This ticket already has evidence checked against its transaction ID; create a new ticket instead.")
    ticket.transaction_id = value
    _save_ticket(ticket, directory)
    return ticket


def _read_document_text(ext: str, data: bytes) -> Optional[str]:
    """Text from a PDF (text layer) or an image (OCR). None when it cannot be read."""
    try:
        if ext == ".pdf":
            import fitz  # PyMuPDF

            with fitz.open(stream=data, filetype="pdf") as doc:
                text = "\n".join(page.get_text("text") for page in doc)
            if text.strip():
                return text[:6000]
            from .ocr import ocr_available, ocr_pdf_pages  # scanned PDF

            if not ocr_available():
                return None
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(data)
            try:
                return "\n".join(t for _, t in ocr_pdf_pages(tmp.name))[:6000]
            finally:
                Path(tmp.name).unlink(missing_ok=True)
        from .ocr import ocr_available, ocr_image_bytes

        return ocr_image_bytes(data)[:6000] if ocr_available() else None
    except Exception:
        return None


def add_attachment(ticket_id: str, user_id: str, filename: str, data: bytes,
                   directory: Optional[str] = None, expected_name: Optional[str] = None) -> Attachment:
    """Attach a screenshot (PNG/JPG) or PDF to the user's own ticket, after verifying it.

    1. The file type is checked from its real content; size and count are limited.
    2. Its text is read (PDF text layer or OCR) and must show the ticket's transaction ID AND the
       user's registered name - otherwise it is rejected (EvidenceRejectedError).
    3. Only then is it saved; OTPs/PINs/card numbers are removed from the stored text.
    """
    ticket = _owned_ticket(ticket_id, user_id, directory)
    if len(ticket.attachments) >= MAX_ATTACHMENTS:
        raise InvalidFileError(f"A ticket can have at most {MAX_ATTACHMENTS} attachments.")
    if not data:
        raise InvalidFileError("The file is empty.")
    if len(data) > MAX_ATTACHMENT_BYTES:
        raise InvalidFileError("The file is too large (limit 5 MB).")
    name = _safe_filename(filename)
    ext = Path(name).suffix.lower()
    if ext not in _MAGIC or not any(data.startswith(sig) for sig in _MAGIC[ext]):
        raise InvalidFileError("Only real PNG, JPG or PDF files can be attached.")
    _require_txn(ticket)
    if expected_name is None:
        from .accounts import get_user

        user = get_user(user_id)
        expected_name = user.full_name if user else ""
    kind = "document" if ext == ".pdf" else "screenshot"

    raw_text = _read_document_text(ext, data)
    checks = verify_document(ticket, raw_text, expected_name, kind)  # raises if it doesn't match

    attachment_id = "A" + secrets.token_hex(4).upper()
    (_attachment_dir(ticket_id, directory) / f"{attachment_id}{ext}").write_bytes(data)
    text, _ = redact_sensitive(raw_text or "")
    text = text.strip()[:2000] or None
    att = Attachment(attachment_id=attachment_id, kind=kind, filename=name, size=len(data),
                     sha256=hashlib.sha256(data).hexdigest(),
                     added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     text=text, hints=extract_evidence_hints(raw_text or ""), verified=True, checks=checks)
    ticket.attachments.append(att)
    _save_ticket(ticket, directory)
    return att


def add_sms_text(ticket_id: str, user_id: str, text: str, directory: Optional[str] = None) -> Attachment:
    """Attach the text of an SMS after verifying it has a date/time and the ticket's transaction ID.

    OTPs, PINs and card numbers are removed before it is saved.
    """
    ticket = _owned_ticket(ticket_id, user_id, directory)
    if len(ticket.attachments) >= MAX_ATTACHMENTS:
        raise InvalidFileError(f"A ticket can have at most {MAX_ATTACHMENTS} attachments.")
    raw = (text or "").strip()[:2000]
    if len(raw) < 5:
        raise InvalidFileError("Paste the SMS text (at least a few words).")
    _require_txn(ticket)
    checks = verify_sms(ticket, raw)  # raises if date or matching transaction ID is missing
    clean, _ = redact_sensitive(raw)
    att = Attachment(attachment_id="A" + secrets.token_hex(4).upper(), kind="sms", filename="sms.txt",
                     size=len(clean.encode("utf-8")), sha256=hashlib.sha256(clean.encode("utf-8")).hexdigest(),
                     added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     text=clean, hints=extract_evidence_hints(raw), verified=True, checks=checks)
    ticket.attachments.append(att)
    _save_ticket(ticket, directory)
    return att


def remove_attachment(ticket_id: str, user_id: str, attachment_id: str, directory: Optional[str] = None) -> None:
    ticket = _owned_ticket(ticket_id, user_id, directory)
    keep = [a for a in ticket.attachments if a.attachment_id != attachment_id]
    if len(keep) == len(ticket.attachments):
        raise PaytmSaathiError("Attachment not found.", "attachment_not_found")
    for path in _attachment_dir(ticket_id, directory).glob(f"{attachment_id}.*"):
        path.unlink(missing_ok=True)
    ticket.attachments = keep
    _save_ticket(ticket, directory)


def apply_hints_to_ticket(ticket_id: str, user_id: str, directory: Optional[str] = None) -> DisputeTicket:
    """Fill empty transaction ID / amount / date on the ticket from what the evidence shows."""
    ticket = _owned_ticket(ticket_id, user_id, directory)
    for att in ticket.attachments:
        h = att.hints
        if not ticket.transaction_id and h.get("utr"):
            ticket.transaction_id = str(h["utr"])
        if not ticket.amount and h.get("amount"):
            ticket.amount = float(h["amount"])
        if not ticket.transaction_date and h.get("date"):
            ticket.transaction_date = str(h["date"])
    _save_ticket(ticket, directory)
    return ticket


def _attachment_lines(ticket: DisputeTicket) -> list[str]:
    if not ticket.attachments:
        return ["Attached: screenshot of the transaction and the debit SMS/statement (to be attached)."]
    lines = ["Supporting evidence attached:"]
    for i, a in enumerate(ticket.attachments, 1):
        label = {"screenshot": "Screenshot", "document": "Document", "sms": "SMS received"}[a.kind]
        verified = f"; checked: {', '.join(a.checks)}" if a.verified and a.checks else ""
        lines.append(f"  {i}. {label}: {a.filename if a.kind != 'sms' else 'text below'} (added {a.added_at[:10]}{verified})")
        if a.kind == "sms" and a.text:
            lines.append("     \"" + " ".join(a.text.split())[:400] + "\"")
    return lines


def build_evidence_pack(ticket: DisputeTicket, customer_name: str, recipient: str = "app",
                        masked_mobile: Optional[str] = None, directory: Optional[str] = None) -> bytes:
    """A ZIP with the complaint letter, every attachment and an index with checksums."""
    letter = draft_complaint(ticket, customer_name, recipient, masked_mobile)
    index = [f"Evidence index for {ticket.ticket_id}", f"Created: {ticket.created_at}", ""]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"complaint_{recipient}.txt", letter)
        folder = Path(directory or DISPUTES_DIR) / "attachments" / ticket.ticket_id
        for a in ticket.attachments:
            if a.kind == "sms":
                zf.writestr(f"evidence/{a.attachment_id}_sms.txt", a.text or "")
            else:
                for path in folder.glob(f"{a.attachment_id}.*"):
                    zf.write(path, f"evidence/{a.attachment_id}_{a.filename}")
            index.append(f"{a.attachment_id}  {a.kind:<10} {a.filename:<30} sha256={a.sha256[:16]}...  added {a.added_at}")
        zf.writestr("evidence_index.txt", "\n".join(index))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Scam / phishing check
# ---------------------------------------------------------------------------
_SCAM_RULES: list[tuple[str, int, str, str]] = [
    (r"\b(otp|one[- ]time (password|pin))\b|ओटीपी", 3, "Asks for or mentions your OTP", "आपका ओटीपी माँगता/बताता है"),
    (r"\b(upi|atm|card)?\s*pin\b|\bcvv\b|पिन", 3, "Asks for a PIN or CVV", "पिन या सीवीवी माँगता है"),
    (r"(anydesk|teamviewer|quicksupport|rustdesk|screen\s*shar)", 4, "Asks you to install a remote-access app", "रिमोट-एक्सेस ऐप इंस्टॉल करने को कहता है"),
    (r"(kyc|pan|aadhaar).{0,40}(expire|expired|update|suspend|block|verify|pending)|(expire|block|suspend).{0,40}kyc|केवाईसी.{0,30}(बंद|ब्लॉक|अपडेट)", 3, "Threatens that KYC/account will be blocked", "केवाईसी/खाता बंद होने की धमकी देता है"),
    (r"(urgent|immediately|within \d+ (hours?|minutes?)|last chance|today only|अभी|तुरंत|जल्दी)", 2, "Creates false urgency", "झूठी जल्दबाज़ी बनाता है"),
    (r"(lottery|lucky draw|you (have )?won|prize|cashback of|reward of|जीत|इनाम|लॉटरी)", 3, "Promises a prize or unexpected reward", "इनाम या अनपेक्षित पैसे का वादा करता है"),
    (r"(collect request|enter (your )?upi pin to receive|receive money.{0,30}pin|पैसे पाने.{0,20}पिन)", 4, "Asks for a UPI PIN to 'receive' money (you never need a PIN to receive)", "पैसे 'पाने' के लिए पिन माँगता है (पैसे पाने के लिए पिन कभी नहीं लगता)"),
    (r"(registration fee|processing fee|advance fee|pay.{0,20}to (release|unlock|get) (loan|refund|prize))", 3, "Asks for an upfront fee to release a loan/refund/prize", "लोन/रिफंड/इनाम पाने के लिए पहले शुल्क माँगता है"),
    (r"(refund|reversal).{0,40}(click|link|http)", 2, "Refund that needs you to click a link", "रिफंड के लिए लिंक पर क्लिक करने को कहता है"),
    # Links, phone numbers and the sender are analysed separately (see linkcheck.py).
]


class ScamAssessment(BaseModel):
    risk: str  # low | medium | high
    score: int
    reasons: list[str]
    reasons_hi: list[str]
    advice: list[str]
    advice_hi: list[str]
    spans: list[tuple[int, int]] = Field(default_factory=list)  # risky phrases to highlight
    links: list[LinkFinding] = Field(default_factory=list)
    phones: list[PhoneFinding] = Field(default_factory=list)
    sender: Optional[SenderFinding] = None
    claimed_brands: list[str] = Field(default_factory=list)
    disclaimer: str = "Automated pattern check; it can miss new scams. When in doubt, do not pay or share anything."


def check_scam_message(text: str, sender: Optional[str] = None) -> ScamAssessment:
    """Risk check of an SMS/WhatsApp/call script: wording, links, phone numbers and sender."""
    if not text or not text.strip():
        raise CalculationInputError("Paste the message you received.")
    score, reasons, reasons_hi, spans = 0, [], [], []
    for pattern, weight, en, hi in _SCAM_RULES:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            score += weight
            reasons.append(en)
            reasons_hi.append(hi)
            spans += [(m.start(), m.end()) for m in matches if m.end() > m.start()]

    brands = detect_brands(text)
    blocklist = load_blocklist()
    links = [analyse_link(u, brands, blocklist) for u in extract_urls(text)[:8]]
    for link in links:
        if link.verdict in ("dangerous", "suspicious"):
            score += 5 if link.verdict == "dangerous" else 2
            reasons.append(f"Link {link.domain or link.url}: {link.reasons[0]}")
            reasons_hi.append(f"लिंक {link.domain or link.url}: {link.reasons_hi[0]}")
            for m in re.finditer(re.escape(link.url), text, re.IGNORECASE):
                spans.append((m.start(), m.end()))
    phones = analyse_phones(text)
    for ph in phones:
        if ph.verdict == "suspicious":
            score += 2
            reasons.append(ph.note)
            reasons_hi.append(ph.note_hi)
    sender_finding = analyse_sender(sender or "", brands)
    if sender_finding and sender_finding.verdict in ("suspicious", "mismatch"):
        score += 3 if sender_finding.verdict == "suspicious" else 2
        reasons.append(sender_finding.note)
        reasons_hi.append(sender_finding.note_hi)

    if re.search(r"(do not|don.t|never)\s+share", text, re.IGNORECASE):
        score = max(score - 3, 0)  # genuine OTP SMS warn the reader not to share
    risk = "high" if score >= 5 else "medium" if score >= 2 else "low"
    if risk == "low" and not reasons:
        advice = ["No common scam patterns found. Still verify the sender through the official app or website."]
        advice_hi = ["कोई आम ठगी का पैटर्न नहीं मिला। फिर भी भेजने वाले को आधिकारिक ऐप या वेबसाइट से जाँचें।"]
    else:
        advice = ["Do not click links, share OTP/PIN, or install apps from this message.",
                  "Verify through the official app or the number printed on your card/statement.",
                  "If you already shared details or paid, call 1930 immediately and block your accounts."]
        advice_hi = ["इस संदेश के लिंक पर क्लिक न करें, ओटीपी/पिन न बताएँ और कोई ऐप इंस्टॉल न करें।",
                     "आधिकारिक ऐप या कार्ड/स्टेटमेंट पर छपे नंबर से पुष्टि करें।",
                     "अगर आप जानकारी बता चुके हैं या पैसे भेज दिए हैं, तुरंत 1930 पर कॉल करें और खाते ब्लॉक करें।"]
    if risk != "low":
        advice.append("Report it: forward the details on the Sanchar Saathi 'Chakshu' page (sancharsaathi.gov.in) and at cybercrime.gov.in.")
        advice_hi.append("रिपोर्ट करें: ब्योरा संचार साथी के 'चक्षु' पेज (sancharsaathi.gov.in) और cybercrime.gov.in पर दें।")
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return ScamAssessment(risk=risk, score=score, reasons=reasons, reasons_hi=reasons_hi,
                          advice=advice, advice_hi=advice_hi, spans=merged, links=links, phones=phones,
                          sender=sender_finding, claimed_brands=brands)
