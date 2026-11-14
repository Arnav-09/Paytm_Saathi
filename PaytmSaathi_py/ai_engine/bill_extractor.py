"""Hospital bill extraction.

Accepts a bill as a text PDF, a scanned PDF, or a photo (PNG/JPG). Text comes from
the PDF text layer or from OCR. Line items are then extracted by

1. the configured LLM (validated with Pydantic), when a real provider is set, and
2. a deterministic rule-based parser otherwise, or if the LLM output is unusable.

Nothing is invented: amounts come only from the bill text. Items that are commonly
not payable (registration, attendant food, etc.) are flagged ``is_payable=False``
for the user to review; the policy, not this module, decides final admissibility.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import fitz  # PyMuPDF

from .exceptions import BillExtractionError, InvalidFileError, LLMProviderError, ValidationError
from .llm_provider import LLMProvider, MockLLMProvider, get_llm_provider
from .ocr import ocr_available, ocr_image_bytes, ocr_pdf_pages
from .schemas import BillLineItem, HospitalBill
from .utils import clean_policy_text, logger

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
BILL_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}

CATEGORIES = [
    "room_rent", "icu_rent", "doctor_fees", "surgery", "nursing", "medicines",
    "investigations", "consumables", "non_payable", "other",
]

# Items commonly excluded by insurers. Flagged for review, not decided here.
NON_PAYABLE_KEYWORDS = [
    "registration", "admission fee", "admission charge", "admin", "file charge",
    "documentation", "attendant", "visitor", "telephone", "tv ", "television",
    "laundry", "toiletr", "cosmetic", "service charge", "surcharge", "diet for attendant",
    "food for attendant", "personal", "convenience", "mortuary", "discharge summary",
    "medical record", "mrd", "courier", "luxury",
]

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("icu_rent", ["icu", "intensive care", "iccu", "nicu", "hdu", "ccu"]),
    ("room_rent", ["room rent", "room charge", "bed charge", "ward", "private room",
                   "semi private", "semi-private", "single room", "deluxe room", "accommodation"]),
    ("surgery", ["surgery", "surgeon", "operation theatre", "operation theater", "ot charge",
                 "ot charges", "theatre", "theater", "anaesthe", "anesthe", "procedure",
                 "implant", "stent", "lens"]),
    ("doctor_fees", ["doctor", "consultant", "consultation", "visit charge", "professional fee",
                     "specialist", "physician", "dr."]),
    ("nursing", ["nursing", "nurse"]),
    ("medicines", ["medicine", "pharmacy", "drug", "injection", "tablet", "iv fluid", "pharma"]),
    ("investigations", ["lab", "pathology", "radiology", "x-ray", "xray", "mri", "ct scan",
                        "ultrasound", "usg", "ecg", "2d echo", "blood test", "investigation",
                        "diagnostic", "scan"]),
    ("consumables", ["consumable", "disposable", "dressing", "syringe", "gloves", "mask",
                     "catheter", "cannula", "kit"]),
]

_SKIP_LINE = re.compile(
    r"(admission|discharge|date|bill\s*(no|number)|invoice|patient|uhid|ip\s*no|mrn|"
    r"phone|mobile|contact|gst|gstin|pan\b|age\b|gender|sex\b|address|doctor name|"
    r"reg(istration)?\s*no|page\s*\d|insurance|policy|tpa|authorization|"
    r"amount in words|rupees)",
    re.IGNORECASE,
)
_TOTAL_LINE = re.compile(
    r"(grand\s*total|net\s*(bill|payable|amount)|total\s*(bill|amount|payable|charges)?|"
    r"gross\s*total|bill\s*total|sub\s*total|subtotal)",
    re.IGNORECASE,
)
_PAYMENT_LINE = re.compile(
    r"(advance|deposit|paid|received|balance|due|refund|discount|concession|round\s*off)",
    re.IGNORECASE,
)
_AMOUNT_AT_END = re.compile(
    r"(?:(?:₹|Rs\.?|INR)\s*)?(?P<amt>\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)\s*(?:/-)?\s*$",
    re.IGNORECASE,
)
_QTY_UNIT = re.compile(
    r"(?P<qty>\d+(?:\.\d+)?)\s*(?:days?|nos?\.?|units?|nights?)?\s*[xX×*@]\s*"
    r"(?:₹|Rs\.?|INR)?\s*(?P<unit>\d[\d,]*(?:\.\d+)?)"
)
_DATE_PATTERNS = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y", "%d %b %Y", "%d %B %Y"]
_DATE_TEXT = r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})"


def to_number(text: str) -> float:
    return float(text.replace(",", ""))


def _has(lowered: str, keyword: str) -> bool:
    """Substring match; short keywords (icu, mri, lab...) must not sit inside longer words."""
    keyword = keyword.strip()
    if len(keyword) <= 4:
        return re.search(rf"(?<![a-z]){re.escape(keyword)}(?![a-z])", lowered) is not None
    return keyword in lowered


def categorize(description: str) -> tuple[str, bool]:
    """Return (category, is_payable) for a bill line description."""
    lowered = description.lower()
    if any(_has(lowered, keyword) for keyword in NON_PAYABLE_KEYWORDS):
        return "non_payable", False
    for category, keywords in _CATEGORY_RULES:
        if any(_has(lowered, keyword) for keyword in keywords):
            return category, True
    return "other", True


def _parse_date(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    cleaned = text.strip()
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def _find(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip(" :-\t") if match else None


def parse_bill_text(text: str, source_filename: Optional[str] = None) -> HospitalBill:
    """Rule-based bill parser (no LLM). Deterministic and offline."""
    text = clean_policy_text(text)
    bill = HospitalBill(source_filename=source_filename, extraction_method="heuristic")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hospital_line = None
    if lines:
        for line in lines[:6]:
            if re.search(r"hospital|clinic|medical cent|nursing home|healthcare|institute", line, re.IGNORECASE):
                bill.hospital_name = line[:120]
                hospital_line = line
                break

    bill.patient_name = _find(r"patient(?:\s*name)?\s*[:\-]\s*([A-Za-z .]{3,60})", text)
    bill.bill_number = _find(r"(?:bill|invoice)\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Za-z0-9/\-]{3,30})", text)
    bill.admission_date = _find(rf"(?:date\s*of\s*)?admission(?:\s*date)?\s*[:\-]?\s*{_DATE_TEXT}", text)
    bill.discharge_date = _find(rf"(?:date\s*of\s*)?discharge(?:\s*date)?\s*[:\-]?\s*{_DATE_TEXT}", text)
    bill.diagnosis = _find(r"(?:diagnosis|provisional diagnosis|ailment)\s*[:\-]\s*([^\n]{3,120})", text)
    bill.procedure = _find(r"(?:procedure|surgery|operation|treatment)\s*(?:name|done|performed)?\s*[:\-]\s*([^\n]{3,120})", text)
    bill.room_type = _find(r"(?:room\s*type|ward\s*type|room\s*category)\s*[:\-]\s*([^\n]{3,40})", text)
    if re.search(r"network\s*hospital|cashless", text, re.IGNORECASE):
        bill.is_network_hospital = True

    admitted, discharged = _parse_date(bill.admission_date), _parse_date(bill.discharge_date)
    if admitted and discharged and discharged >= admitted:
        bill.days_of_stay = max((discharged - admitted).days, 1)

    items: list[BillLineItem] = []
    for line in lines:
        if line == hospital_line:
            continue
        if _TOTAL_LINE.search(line):
            amount = _AMOUNT_AT_END.search(line)
            if (
                amount
                and not _PAYMENT_LINE.search(line)
                and re.search(r"grand|net|total", line, re.IGNORECASE)
                and not re.search(r"sub\s*total", line, re.IGNORECASE)
            ):
                bill.stated_total = to_number(amount.group("amt"))
            continue
        if _SKIP_LINE.search(line) or _PAYMENT_LINE.search(line):
            continue
        amount = _AMOUNT_AT_END.search(line)
        if not amount:
            continue
        description = line[: amount.start()].strip(" :-.₹\t|")
        description = re.sub(r"^\d{1,3}[.)]\s+", "", description)  # serial number
        if len(description) < 3 or not re.search(r"[A-Za-z]{3}", description):
            continue
        value = to_number(amount.group("amt"))
        if value <= 0:
            continue
        item = BillLineItem(description=description, amount=value)
        qty_match = _QTY_UNIT.search(description)
        if qty_match:
            item.quantity = to_number(qty_match.group("qty"))
            item.unit_price = to_number(qty_match.group("unit"))
        item.category, item.is_payable = categorize(description)
        if not item.is_payable:
            item.note = "Commonly non-payable; verify against your policy."
        items.append(item)

    bill.line_items = items
    bill.computed_total = round(sum(i.amount for i in items), 2)

    if bill.days_of_stay is None:
        for item in items:
            if item.category == "room_rent" and item.quantity:
                bill.days_of_stay = int(item.quantity)
                break

    _add_warnings(bill)
    return bill


def _add_warnings(bill: HospitalBill) -> None:
    if not bill.line_items:
        bill.warnings.append("No line items could be read. Please add them manually.")
    if bill.stated_total and bill.computed_total:
        gap = abs(bill.stated_total - bill.computed_total)
        if gap > max(1.0, 0.02 * bill.stated_total):
            bill.warnings.append(
                f"Line items add up to ₹{bill.computed_total:,.2f} but the bill total is "
                f"₹{bill.stated_total:,.2f}. Some items may be missing or misread - please review."
            )
    if bill.days_of_stay is None:
        bill.warnings.append("Length of stay not found. Enter it to check the room rent limit.")
    if bill.used_ocr:
        bill.warnings.append("Text was read by OCR. Please double-check the amounts.")


_LLM_SCHEMA = {
    "hospital_name": "string|null", "patient_name": "string|null", "bill_number": "string|null",
    "admission_date": "string|null", "discharge_date": "string|null", "days_of_stay": "int|null",
    "diagnosis": "string|null", "procedure": "string|null", "room_type": "string|null",
    "stated_total": "number|null",
    "line_items": [{"description": "string", "category": "|".join(CATEGORIES),
                    "quantity": "number|null", "unit_price": "number|null", "amount": "number"}],
}


def _llm_extract(text: str, llm: LLMProvider, source_filename: Optional[str]) -> HospitalBill:
    prompt = (
        "Extract this Indian hospital bill into JSON. Use ONLY information present in the text. "
        "Use null for anything missing. Never invent items or amounts. Do not include totals, "
        "advances, discounts or taxes as line items.\n\nBILL TEXT:\n" + text[:12000]
    )
    data = llm.generate_structured(prompt, _LLM_SCHEMA)
    if not isinstance(data, dict) or not data.get("line_items"):
        raise ValidationError("LLM returned no line items.")
    try:
        bill = HospitalBill.model_validate({**data, "extraction_method": "llm", "source_filename": source_filename})
    except Exception as exc:
        raise ValidationError(f"LLM bill output failed validation: {exc}") from exc
    for item in bill.line_items:
        if item.category not in CATEGORIES:
            item.category = "other"
        category, payable = categorize(item.description)
        if not payable:  # keyword safety net regardless of LLM judgement
            item.category, item.is_payable = category, False
            item.note = "Commonly non-payable; verify against your policy."
    bill.computed_total = round(sum(i.amount for i in bill.line_items), 2)
    return bill


def extract_bill_from_text(
    text: str,
    source_filename: Optional[str] = None,
    llm: Optional[LLMProvider] = None,
    used_ocr: bool = False,
) -> HospitalBill:
    if not text or not text.strip():
        raise BillExtractionError("The bill contains no readable text.")
    llm = llm if llm is not None else get_llm_provider()
    bill: Optional[HospitalBill] = None
    if not isinstance(llm, MockLLMProvider):
        try:
            bill = _llm_extract(text, llm, source_filename)
        except (LLMProviderError, ValidationError) as exc:
            logger.warning("LLM bill extraction failed (%s); using rule-based parser.", exc)
    if bill is None:
        bill = parse_bill_text(text, source_filename)
        bill.used_ocr = used_ocr
        bill.warnings = []
        _add_warnings(bill)
        return bill
    bill.used_ocr = used_ocr
    _add_warnings(bill)
    return bill


def extract_bill(
    file_path: Union[str, Path], llm: Optional[LLMProvider] = None
) -> HospitalBill:
    """Extract a structured hospital bill from a PDF or image file."""
    path = Path(file_path)
    if not path.exists():
        raise InvalidFileError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in BILL_EXTENSIONS:
        raise InvalidFileError("Bills must be a PDF or an image (PNG, JPG, JPEG).")
    if path.stat().st_size == 0:
        raise InvalidFileError("The uploaded bill file is empty.")

    used_ocr = False
    if suffix in IMAGE_EXTENSIONS:
        text = ocr_image_bytes(path.read_bytes())  # raises OCRUnavailableError if no OCR
        used_ocr = True
    else:
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise BillExtractionError(f"Could not open the bill PDF: {exc}") from exc
        try:
            text = "\n".join((page.get_text("text") or "") for page in doc)
        finally:
            doc.close()
        if not text.strip():
            if not ocr_available():
                raise BillExtractionError(
                    "This bill PDF is a scan and OCR is not installed. "
                    "Run: pip install rapidocr-onnxruntime"
                )
            text = "\n".join(page_text for _, page_text in ocr_pdf_pages(path))
            used_ocr = True
    return extract_bill_from_text(text, source_filename=path.name, llm=llm, used_ocr=used_ocr)
