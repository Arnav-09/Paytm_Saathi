"""Bank / credit-card statement ingestion: CSV, TSV, Excel or PDF -> normalised, masked transactions.

Layout-tolerant by design, because every bank exports differently:
  * the transaction table is found anywhere in the first 60 rows (account details, summaries and blank lines before
    it are fine) and columns are matched by meaning ("Txn Date", "Value Dt", "Narration", "Withdrawal Amt. (INR)",
    "Dr/Cr" ...), not by position; extra columns are ignored, several description columns are combined;
  * files without a header row get their columns guessed from the values (dates, amounts, text);
  * any delimiter (comma, semicolon, tab, pipe), UTF-8 / UTF-16 / Windows encodings, every sheet of a workbook,
    HTML or XML "Excel" exports some banks send as .xls;
  * dates as text in most common styles (01/11/2026, 2026-11-01, 1 Nov 2026, Nov 01, 2026, 01-Nov-26, with times)
    or as real Excel dates; amounts with Rs/INR/₹, Indian digit grouping, Cr/Dr markers, brackets or minus signs;
  * one signed amount column in either convention (bank: minus = money out; card export: minus = payment/refund);
  * repeated header rows (page breaks), descriptions wrapped onto a second row, opening/closing balance and total rows;
  * PDFs: rows are rebuilt from the position of each word on the page (works for drawn tables whose cells come out
    as separate text lines) with a line-pattern fallback; summary values next to or below their labels.

Every transaction keeps a ``source`` reference (``row 14`` for spreadsheets, ``page 2, line 7`` for PDFs)
so each insight can point back to the exact place in the user's file.

Privacy: descriptions are masked on read - card numbers, Aadhaar/PAN, OTPs and long account numbers
are replaced (last 4 digits kept) before anything is analysed or shown. Nothing is written to disk here.
"""

from __future__ import annotations

import io
import json
import numbers
import re
import unicodedata
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .exceptions import InvalidFileError, PaytmSaathiError
from .guardrails import redact_sensitive

RULES_FILE = Path(__file__).resolve().parent.parent / "knowledge" / "money_rules.json"
MAX_STATEMENT_BYTES = 10 * 1024 * 1024
MAX_TRANSACTIONS = 5000
_MAGIC = {".pdf": b"%PDF-", ".xlsx": b"PK\x03\x04"}
_HEADER_SEARCH_ROWS = 60


class StatementError(PaytmSaathiError):
    def __init__(self, message: str):
        super().__init__(message, "statement_error")


class Transaction(BaseModel):
    txn_id: str
    date: str  # YYYY-MM-DD
    description: str  # masked
    merchant: str
    merchant_key: str
    amount: float  # always positive
    type: str  # debit | credit
    category: str = "Other"
    recurring: bool = False
    duplicate: bool = False
    fee: bool = False
    balance: Optional[float] = None
    source: str  # "file.csv row 14" / "file.pdf page 2, line 7"
    period: Optional[str] = None  # billing period (YYYY-MM of the bill) for card statements; else calendar month


class Statement(BaseModel):
    source_filename: str
    kind: str = "bank"  # bank | card
    transactions: list[Transaction] = Field(default_factory=list)
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    card_total_due: Optional[float] = None
    card_minimum_due: Optional[float] = None
    card_due_date: Optional[str] = None
    card_credit_limit: Optional[float] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def _kw_regex(keyword: str) -> re.Pattern:
    # whole word/phrase, plural allowed ("finance charge" matches "FINANCE CHARGES")
    return re.compile(rf"(?<![a-z0-9]){re.escape(keyword.strip().lower())}(?:s|es)?(?![a-z0-9])")


@lru_cache(maxsize=1)
def _category_patterns() -> list[tuple[str, list[tuple[str, re.Pattern]]]]:
    return [(name, [(kw, _kw_regex(kw)) for kw in kws]) for name, kws in rules()["categories"]]


@lru_cache(maxsize=1)
def _credit_patterns() -> list[re.Pattern]:
    return [_kw_regex(k) for k in rules()["credit_keywords"]]


_BRAND_CATEGORIES = {"Subscriptions", "Insurance", "Food & dining", "Groceries", "Shopping", "Travel & fuel",
                     "Bills & utilities", "Health", "Investments", "Education", "Rent & housing"}
_GENERIC_TOKENS = {
    "UPI", "POS", "NEFT", "IMPS", "RTGS", "ACH", "NACH", "ECS", "DR", "CR", "TO", "FROM", "BY", "TRF", "TRANSFER",
    "PAYMENT", "PAY", "TXN", "REF", "INB", "MB", "MOB", "IB", "BIL", "BILLPAY", "PURCHASE", "DEBIT", "CREDIT", "CARD",
    "VPS", "IPS", "ONLINE", "WWW", "COM", "IN", "PVT", "LTD", "LIMITED", "INDIA", "THE", "AND", "XXXX", "P2M", "P2A",
    "ATW", "EAW", "NWD", "SI", "AUTOPAY", "MANDATE", "OK", "OKAXIS", "OKHDFCBANK", "OKICICI", "OKSBI", "YBL", "PAYTM",
}
# Card statements: these are money coming back onto the card (payments, refunds), not spending.
_CARD_CREDIT = re.compile(r"payment\s+received|thank\s*you|refund|reversal|cashback|cash\s+back|credit\s+adj", re.I)


def categorise(description: str, txn_type: str) -> tuple[str, Optional[str]]:
    """(category, matched brand keyword). Credits only become Income / Refunds / Money in."""
    text = description.lower()
    for name, patterns in _category_patterns():
        if txn_type == "credit" and name not in ("Income", "Refunds & cashback"):
            continue
        hits = [kw for kw, pat in patterns if pat.search(text)]
        if hits:
            brand = max(hits, key=len)  # most specific: "lic of india" beats "premium"
            return name, (brand if name in _BRAND_CATEGORIES else None)
    return ("Money in" if txn_type == "credit" else "Other"), None


def merchant_of(description: str, brand: Optional[str]) -> tuple[str, str]:
    """(display name, grouping key)."""
    if brand:
        return brand.strip().title(), brand.strip().lower()
    tokens = re.split(r"[\s/\-\*_:@#|.,()]+", description.upper())
    words = [t for t in tokens if len(t) > 1 and not any(c.isdigit() for c in t) and t not in _GENERIC_TOKENS]
    if not words:
        return description[:24].strip() or "Unknown", (description[:24].strip() or "unknown").lower()
    return " ".join(words[:2]).title(), words[0].lower()


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
_MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december"]
_MONTHS = {**{m: i for i, m in enumerate(_MONTH_NAMES, 1)}, **{m[:3]: i for i, m in enumerate(_MONTH_NAMES, 1)},
           "sept": 9}
_DATE_RX = (r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
            r"|\d{1,2}[-/.]\d{1,2}[-/.](?:\d{4}|\d{2})"
            r"|\d{1,2}(?:st|nd|rd|th)?[\s\-/.]*[A-Za-z]{3,9}\.?[\s\-/.,']*(?:\d{4}|\d{2})"
            r"|[A-Za-z]{3,9}\.?[\s\-]+\d{1,2}(?:st|nd|rd|th)?,?[\s\-]+(?:\d{4}|\d{2})")
_DATE_TOKEN = re.compile(rf"(?<!\w)(?:{_DATE_RX})(?!\d)", re.I)
_TIME = r"(?:[T\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:\s*[AaPp][Mm])?)?"


def _year(text: str) -> int:
    y = int(text)
    return y if y >= 100 else (2000 + y if y < 70 else 1900 + y)


def _mk_date(y: int, m: Optional[int], d: int) -> Optional[date]:
    if not m or not 1990 <= y <= 2100:
        return None
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _date_from_token(tok: str) -> Optional[date]:
    tok = tok.strip()
    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", tok):
        return _mk_date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})", tok):
        d, mo = int(m[1]), int(m[2])
        if mo > 12 and d <= 12:  # US style 11/25/2026
            d, mo = mo, d
        return _mk_date(_year(m[3]), mo, d)
    if m := re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/.]*([A-Za-z]{3,9})\.?[\s\-/.,']*(\d{4}|\d{2})", tok, re.I):
        return _mk_date(_year(m[3]), _MONTHS.get(m[2].lower()), int(m[1]))
    if m := re.fullmatch(r"([A-Za-z]{3,9})\.?[\s\-]+(\d{1,2})(?:st|nd|rd|th)?,?[\s\-]+(\d{4}|\d{2})", tok, re.I):
        return _mk_date(_year(m[3]), _MONTHS.get(m[1].lower()), int(m[2]))
    return None


def _yyyymmdd(text: str) -> Optional[date]:
    return _mk_date(int(text[:4]), int(text[4:6]), int(text[6:]))


def parse_date(value, *, allow_serial: bool = True) -> Optional[date]:
    """Date from a cell: real date/datetime, Excel serial number, yyyymmdd, or text in most common styles.

    Day-first for numeric dates (Indian style); a date like 11/25/2026 that only makes sense month-first is swapped.
    A date inside longer text ("01/11/2026 10:32:05", "Txn on 1st Nov 2026") is found too.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):  # includes pandas Timestamp
        try:
            return value.date() if value == value else None
        except (ValueError, TypeError):
            return None
    if isinstance(value, date):
        return value
    if isinstance(value, numbers.Real):
        v = float(value)
        if v != v:
            return None
        if allow_serial and 32874 <= v <= 73051:  # Excel serial day number (1990-2100)
            return date(1899, 12, 30) + timedelta(days=int(v))
        if v.is_integer() and 19900101 <= v <= 21001231:
            return _yyyymmdd(str(int(v)))
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text or len(text) > 80:
        return None
    if re.fullmatch(r"\d{8}", text):
        return _yyyymmdd(text)
    if allow_serial and re.fullmatch(r"\d{5}", text):
        return parse_date(float(text))
    for m in _DATE_TOKEN.finditer(text):
        if d := _date_from_token(m.group(0)):
            return d
    return None


# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------
_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", " ": " ", " ": " ", " ": " "})
_CUR = r"(?:₹|rs\.?|inr|rupees)"
_MONEY_RX = re.compile(
    rf"(?:(?P<mk0>cr|dr)\.?\s*)?(?P<s1>[+-])?\s*{_CUR}?\s*(?P<s2>[+-])?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d+)?|\.\d+)"
    rf"\s*(?P<s3>-)?\s*{_CUR}?\s*(?P<mk>cr|dr|credit|debit)?\.?\s*(?P<s4>-)?", re.I)
_NUM_OK = re.compile(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+")
_MARKERS = {"cr": "cr", "credit": "cr", "dr": "dr", "debit": "dr"}


def _money(value) -> tuple[Optional[float], Optional[str]]:
    """(signed amount, marker) where marker is 'cr', 'dr', 'neg' (minus sign / brackets) or None."""
    if value is None or isinstance(value, bool):
        return None, None
    if isinstance(value, numbers.Real):
        v = float(value)
        return (None, None) if v != v else (v, "neg" if v < 0 else None)
    text = unicodedata.normalize("NFKC", str(value)).translate(_DASHES).strip()
    if not text or text.lower() in {"-", "--", "nil", "na", "n/a", "none"}:
        return None, None
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative, text = True, text[1:-1].strip()
    m = _MONEY_RX.fullmatch(text)
    if not m or not _NUM_OK.fullmatch(m["num"]):
        return None, None
    amount = float(m["num"].replace(",", ""))
    marker = _MARKERS.get((m["mk"] or m["mk0"] or "").lower())
    negative = negative or "-" in (m["s1"] or "", m["s2"] or "", m["s3"] or "", m["s4"] or "")
    if marker is None and negative:
        marker = "neg"
    return (-amount if marker in ("dr", "neg") else amount), marker


def parse_money(value) -> Optional[float]:
    """'1,234.50' / '₹1,234.50 Dr' / '(500.00)' / '-200' / '200-' / 'INR 1,00,000' -> signed float
    (Dr, brackets and minus signs are negative; Cr is positive)."""
    return _money(value)[0]


_LONG_NUMBER = re.compile(r"(?<![\d.,])\d{9,}(?![\d.,])")


def mask_description(text: str) -> str:
    text, _ = redact_sensitive(re.sub(r"\s+", " ", str(text or "")).strip())
    return _LONG_NUMBER.sub(lambda m: "XXXX" + m.group(0)[-4:], text)[:160]


def _guess_type(description: str) -> str:
    low = description.lower()
    return "credit" if any(p.search(low) for p in _credit_patterns()) else "debit"


def _credit_like(description: str) -> bool:
    return _guess_type(description) == "credit" or bool(_CARD_CREDIT.search(description))


def _type_value(value) -> Optional[str]:
    """'CR' / 'C' / 'Credit' / 'Deposit' -> credit; 'DR' / 'D' / 'Debit' / 'Withdrawal' -> debit; else None."""
    t = re.sub(r"[^a-z+\-]", "", str(value or "").lower())
    if t in {"cr", "c", "credit", "deposit", "in", "+", "received", "inward"} or t.startswith("credit"):
        return "credit"
    if t in {"dr", "d", "debit", "withdrawal", "out", "-", "paid", "outward"} or t.startswith("debit"):
        return "debit"
    return None


# ---------------------------------------------------------------------------
# Column matching
# ---------------------------------------------------------------------------
_COLS = {  # field -> header names in priority order (after _norm)
    "date": ["txn date", "transaction date", "tran date", "trans date", "trxn date", "posting date", "post date",
             "booking date", "date", "txn dt", "tran dt", "value date", "value dt", "date time", "datetime",
             "transaction datetime", "dt"],
    "desc": ["description", "narration", "particulars", "transaction details", "transaction description",
             "transaction particulars", "transaction remarks", "txn details", "txn description", "details", "remarks",
             "merchant name", "merchant", "payee", "beneficiary", "memo", "info"],
    "type": ["dr/cr", "cr/dr", "debit/credit", "credit/debit", "dr cr", "cr dr", "d/c", "c/d", "dr or cr",
             "txn type", "transaction type", "tran type", "type", "indicator"],
    "debit": ["withdrawal amt", "withdrawal amount", "withdrawals", "withdrawal", "withdrawl", "debit amount",
              "debit amt", "debits", "debit", "dr amount", "dr amt", "dr", "money out", "paid out", "outflow", "spent"],
    "credit": ["deposit amt", "deposit amount", "deposits", "deposit", "credit amount", "credit amt", "credits",
               "credit", "cr amount", "cr amt", "cr", "money in", "paid in", "inflow", "received"],
    "amount": ["transaction amount", "txn amount", "tran amount", "amount", "amt", "billing amount"],
    "balance": ["closing balance", "running balance", "available balance", "balance amount", "balance",
                "closing bal", "available bal", "avl bal", "bal"],
}
_MONEY_FIELDS = {"amount", "debit", "credit"}


def _norm(header) -> str:
    """'Withdrawal Amt. (INR )' -> 'withdrawal amt'; 'Dr / Cr' -> 'dr/cr'; 'Date & Time' -> 'date time'."""
    text = unicodedata.normalize("NFKC", str(header if header is not None else "")).lower()
    text = re.sub(r"[^a-z/]+", " ", text)
    words = [w for w in text.split() if w not in {"inr", "rs", "rupees"}]
    while words and words[-1] in {"in", "of"}:
        words.pop()
    return re.sub(r"\s*/\s*", "/", " ".join(words)).strip("/ ")


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[ /]+", text) if t]


def _contains(header: str, name: str) -> bool:
    ht, nt = _tokens(header), _tokens(name)
    return bool(nt) and any(ht[i:i + len(nt)] == nt for i in range(len(ht) - len(nt) + 1))


_NOT_A_COLUMN = {"limit", "count", "points", "outstanding", "due", "no", "number", "id", "rate", "reward", "minimum",
                 "total", "card", "account", "cheque", "chq", "ref", "category", "mode", "branch", "code"}


def _map_columns(headers: list) -> dict:
    """field -> column index. Exact header names win over names that only contain a synonym; a header such as
    "Credit Limit", "Dr Count" or "Payment Due Date" never partly matches Credit / Debit / Date."""
    norm = [_norm(h) for h in headers]
    mapping: dict[str, int] = {}
    for exact in (True, False):
        for field, names in _COLS.items():
            if field in mapping:
                continue
            for name in names:
                hit = next((i for i, h in enumerate(norm) if h and i not in mapping.values()
                            and (h == name if exact else (_contains(h, name) and not _NOT_A_COLUMN & set(_tokens(h))))),
                           None)
                if hit is not None:
                    mapping[field] = hit
                    break
    return mapping


def _extra_desc_columns(headers: list, mapping: dict) -> list[int]:
    """Other description-like columns ("Remarks", "Merchant") to append to the main description."""
    used = set(mapping.values())
    return [i for i, h in enumerate(headers) if i not in used and _norm(h)
            and any(_contains(_norm(h), n) for n in _COLS["desc"])]


def _is_header(row: list) -> Optional[dict]:
    if sum(1 for c in row if str(c).strip()) < 3:
        return None
    m = _map_columns(row)
    if "date" not in m or not (_MONEY_FIELDS & m.keys()):
        return None
    if "desc" not in m:  # no recognised description column: use the first unnamed-but-filled text column
        free = [i for i, c in enumerate(row) if str(c).strip() and i not in m.values()
                and parse_money(c) is None and parse_date(c, allow_serial=False) is None]
        if not free:
            return None
        m["desc"] = free[0]
    return m


def _infer_columns(rows: list[list]) -> Optional[dict]:
    """No header row: guess columns from the values (a date column, amount columns, the longest text column)."""
    sample = [r for r in rows if sum(1 for c in r if str(c).strip()) >= 3][:200]
    if len(sample) < 3:
        return None
    width = max(len(r) for r in sample)

    def col(i):
        return [r[i] if i < len(r) else "" for r in sample]

    def filled(vals):
        return [v for v in vals if str(v).strip()]

    stats = []
    for i in range(width):
        vals = filled(col(i))
        n = len(vals) or 1
        dates = sum(1 for v in vals if parse_date(v, allow_serial=False))
        moneyish = sum(1 for v in vals if parse_money(v) is not None and parse_date(v, allow_serial=False) is None
                       and (isinstance(v, float) or re.search(r"[.,]", str(v)) or len(str(v).strip()) <= 7))
        types = sum(1 for v in vals if _type_value(v))
        text_len = sum(len(str(v)) for v in vals if re.search(r"[A-Za-z]{3}", str(v)) and not parse_date(v)) / n
        stats.append({"i": i, "fill": len(vals) / len(sample), "date": dates / n, "money": moneyish / n,
                      "type": types / n, "text": text_len})
    date_cols = [s for s in stats if s["date"] >= 0.8 and s["fill"] >= 0.8]
    if not date_cols:
        return None
    mapping = {"date": date_cols[0]["i"]}
    money_cols = [s for s in stats if s["i"] != mapping["date"] and s["money"] >= 0.8 and s["date"] < 0.5]
    if not money_cols:
        return None
    type_col = next((s for s in stats if s["type"] >= 0.9 and s["i"] not in mapping.values() and s["money"] < 0.5), None)
    if type_col:
        mapping["type"] = type_col["i"]
    dense = [s for s in money_cols if s["fill"] >= 0.9]
    if len(money_cols) == 1:
        mapping["amount"] = money_cols[0]["i"]
    elif len(money_cols) == 2 and len(dense) == 2:
        mapping["amount"], mapping["balance"] = money_cols[0]["i"], money_cols[1]["i"]
    elif len(money_cols) == 2:
        mapping["debit"], mapping["credit"] = money_cols[0]["i"], money_cols[1]["i"]
    else:
        last3 = money_cols[-3:]
        if last3[2]["fill"] >= 0.9 and last3[0]["fill"] < 0.9 and last3[1]["fill"] < 0.9:
            mapping["debit"], mapping["credit"], mapping["balance"] = (s["i"] for s in last3)
        else:
            mapping["amount"], mapping["balance"] = money_cols[0]["i"], money_cols[-1]["i"]
    text_cols = sorted((s for s in stats if s["i"] not in mapping.values() and s["text"] > 0),
                       key=lambda s: -s["text"])
    if not text_cols:
        return None
    mapping["desc"] = text_cols[0]["i"]
    return mapping


# ---------------------------------------------------------------------------
# Table -> statement (shared by CSV, Excel and PDF tables)
# ---------------------------------------------------------------------------
_OPEN_ROW = re.compile(r"opening\s+bal|balance\s+(?:b/?f|brought\s+forward)|brought\s+forward|^\s*b/?f\s*$", re.I)
_CLOSE_ROW = re.compile(r"closing\s+bal|balance\s+(?:c/?f|carried\s+forward)|carried\s+forward|^\s*c/?f\s*$", re.I)
_TOTAL_ROW = re.compile(r"^\s*(?:grand\s+|sub\s*-?\s*)?totals?\b|^\s*statement\s+summary", re.I)
_NOT_WRAP = re.compile(r"end\s+of\s+statement|computer\s+generated|page\s+\d+|continued|\bsummary\b|\bnote\b"
                       r"|abbreviation|generated\s+on|^\*+", re.I)
_CARD_TEXT = re.compile(r"credit\s+card|card\s+(?:no|number|statement|account)|payment\s+due\s+date|"
                        r"minimum\s+(?:amount\s+)?due|min(?:imum)?\.?\s+amt\.?\s+due|total\s+outstanding", re.I)


def _cell_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:
        return ""
    return str(v).strip()


def _is_card_hint(filename: str, text: str) -> bool:
    return bool(re.search(r"(?<![a-z])(?:credit[\s_\-]?)?card(?![a-z])", filename.lower()) or _CARD_TEXT.search(text))


def _rows_statement(rows: list[list], header_idx: int, mapping: dict, extras: list[int], sources: list,
                    filename: str, card_hint: bool) -> tuple[Statement, list[list]]:
    """Read the rows under a header. Returns the statement and the leftover rows (summary blocks, footers)."""
    stmt = Statement(source_filename=filename)
    pending: list[dict] = []
    leftovers: list[list] = []
    header = rows[header_idx] if header_idx >= 0 else None
    money_idx = {mapping[f] for f in ("amount", "debit", "credit", "balance") if f in mapping}
    desc_idx = {mapping["desc"], *extras}
    last_was_txn = False
    prev_balance: Optional[float] = None

    for i in range(header_idx + 1, len(rows)):
        row = rows[i]
        cells = [_cell_text(c) for c in row]
        if not any(cells):
            last_was_txn = False
            continue

        def cell(field, _row=row):
            idx = mapping.get(field)
            return _row[idx] if idx is not None and idx < len(_row) else ""

        joined = " ".join(c for c in cells if c)
        if header is not None and _norm(cell("date")) and _norm(cell("date")) == _norm(header[mapping["date"]]):
            last_was_txn = False  # header repeated after a page break
            continue
        desc_raw = " ".join(cells[j] for j in [mapping["desc"], *extras] if j < len(cells) and cells[j])
        d = parse_date(cell("date"))
        label = desc_raw or joined
        if _OPEN_ROW.search(label) or _CLOSE_ROW.search(label):
            values = [parse_money(cell("balance"))] + [parse_money(c) for c in reversed(cells)]
            value = next((v for v in values if v is not None and parse_date(v, allow_serial=False) is None), None)
            if value is None:
                leftovers.append(row)  # a summary block: labels here, values in the row below
            elif _OPEN_ROW.search(label) and stmt.opening_balance is None:
                stmt.opening_balance = value
                prev_balance = value
            elif _CLOSE_ROW.search(label):
                stmt.closing_balance = value
            last_was_txn = False
            continue
        if d is None:
            only_desc = all(not c or j in desc_idx for j, c in enumerate(cells))
            if (last_was_txn and pending and only_desc and desc_raw and len(desc_raw) <= 90
                    and not _NOT_WRAP.search(desc_raw) and not _TOTAL_ROW.search(desc_raw)):
                pending[-1]["desc"] += " " + desc_raw  # description wrapped onto the next row
            else:
                leftovers.append(row)
                last_was_txn = False
            continue
        if _TOTAL_ROW.search(desc_raw):
            leftovers.append(row)
            last_was_txn = False
            continue
        debit, _ = _money(cell("debit"))
        credit, _ = _money(cell("credit"))
        amount, marker = _money(cell("amount"))
        balance = parse_money(cell("balance"))
        entry = {"date": d, "desc": desc_raw, "balance": balance, "source": sources[i], "sign": None}
        if debit:
            entry.update(type="debit", amount=abs(debit))
        elif credit:
            entry.update(type="credit", amount=abs(credit))
        elif amount:
            t = {"cr": "credit", "dr": "debit"}.get(marker) or _type_value(cell("type"))
            entry.update(type=t, amount=abs(amount))
            if t is None:
                if balance is not None and prev_balance is not None:  # running balance tells the direction
                    if abs(prev_balance - abs(amount) - balance) < 0.01:
                        entry["type"] = "debit"
                    elif abs(prev_balance + abs(amount) - balance) < 0.01:
                        entry["type"] = "credit"
                if entry["type"] is None:
                    entry["sign"] = "neg" if amount < 0 else "pos"
        else:
            leftovers.append(row)
            last_was_txn = False
            continue
        if balance is not None:
            prev_balance = balance
        pending.append(entry)
        last_was_txn = True

    _resolve_signs(pending, card_hint)
    for p in pending:
        _add(stmt, p["date"], p["desc"], p["amount"], p["type"], p["balance"], p["source"])
    return stmt, leftovers


def _resolve_signs(pending: list[dict], card_hint: bool) -> None:
    """One amount column without Dr/Cr: decide which sign convention the file uses.

    Bank exports: minus = money out. Card exports: plus = spend, minus = payment/refund. The descriptions decide
    (payments and refunds on the minus side -> card style); a tie falls back to whether this looks like a card file.
    """
    open_rows = [p for p in pending if p["type"] is None]
    if not open_rows:
        return
    negatives = [p for p in open_rows if p["sign"] == "neg"]
    if negatives:
        neg_credit = sum(1 for p in negatives if _credit_like(p["desc"]))
        pos_credit = sum(1 for p in open_rows if p["sign"] == "pos" and _credit_like(p["desc"]))
        card_style = neg_credit > pos_credit or (neg_credit == pos_credit and card_hint)
        for p in open_rows:
            if card_style:
                p["type"] = "credit" if p["sign"] == "neg" else "debit"
            else:
                p["type"] = "debit" if p["sign"] == "neg" else ("credit" if pos_credit else _guess_type(p["desc"]))
        return
    for p in open_rows:  # all positive: go by the words
        p["type"] = "credit" if (_credit_like(p["desc"]) if card_hint else _guess_type(p["desc"]) == "credit") else "debit"


def _parse_table(rows: list[list], filename: str, sources: Optional[list] = None, hint_text: str = "",
                 meta_lines: Optional[list] = None, card_hint: Optional[bool] = None) -> Statement:
    """Find the transaction table in a grid of cells and read it. ``sources`` gives each row's reference."""
    if sources is None:
        sources = [f"{filename} row {i + 1}" for i in range(len(rows))]
    preamble_all = " ".join(_cell_text(c) for r in rows[:_HEADER_SEARCH_ROWS] for c in r)
    hint = _is_card_hint(filename, hint_text + " " + preamble_all) if card_hint is None else card_hint
    best: Optional[tuple[Statement, list[list], int]] = None
    for idx, row in enumerate(rows[:_HEADER_SEARCH_ROWS]):
        mapping = _is_header(row)
        if not mapping:
            continue
        stmt, leftovers = _rows_statement(rows, idx, mapping, _extra_desc_columns(row, mapping), sources, filename, hint)
        if best is None or len(stmt.transactions) > len(best[0].transactions):
            best = (stmt, leftovers, idx)
    if best is None or not best[0].transactions:
        mapping = _infer_columns(rows)
        if mapping:
            stmt, leftovers = _rows_statement(rows, -1, mapping, [], sources, filename, hint)
            if stmt.transactions:
                stmt.warnings.append("This file has no column names, so the columns were guessed from the values. "
                                     "Please check a few amounts.")
                best = (stmt, leftovers, -1)
    if best is None or not best[0].transactions:
        raise StatementError("Could not find the transaction table. The file needs a date, a description and an "
                             "amount (or debit/credit) for each transaction.")
    stmt, leftovers, idx = best
    lines = meta_lines if meta_lines is not None else _grid_lines(rows[:max(idx, 0)]) + _grid_lines(leftovers)
    _read_summary(stmt, lines, hint_text, period=hint)
    if hint:
        stmt.kind = "card"
    return stmt


def _grid_lines(rows: list[list]) -> list[list[tuple]]:
    """Spreadsheet rows as positioned lines (column index as x) for the summary reader."""
    return [[(j * 10, j * 10 + 6, _cell_text(c)) for j, c in enumerate(r) if _cell_text(c)]
            for r in rows if any(_cell_text(c) for c in r)]


# ---------------------------------------------------------------------------
# Summary values (card due date / totals / limits / balances / period)
# ---------------------------------------------------------------------------
_AMT_VALUE = r"(?:\(?\s*(?:₹|rs\.?|inr)\s*\)?)?\s*(\d[\d,]*(?:\.\d+)?)(?![\d,.]*[/\-]\d)"
_LABELS = {
    "card_due_date": r"(?:payment\s+)?due\s+(?:date|by)|pay\s+by(?:\s+date)?",
    "card_minimum_due": r"min(?:imum)?\.?\s*(?:amount|amt\.?|payment)?\s*(?:due|payable)|minimum\s+payment",
    "card_total_due": r"total\s+(?:amount\s+)?(?:due|payable|outstanding)|total\s+payment\s+due|"
                      r"(?<!minimum\s)(?<!min\.\s)(?<!min\s)amount\s+payable|new\s+balance|statement\s+balance|"
                      r"current\s+outstanding",
    "card_credit_limit": r"(?<!available\s)(?<!cash\s)(?:total\s+)?credit\s+limit|card\s+limit",
    "opening_balance": r"opening\s+bal(?:ance)?|balance\s+brought\s+forward|balance\s+b/f",
    "closing_balance": r"closing\s+bal(?:ance)?|balance\s+carried\s+forward|balance\s+c/f",
}
_INLINE = {f: re.compile(rf"(?:{lab})\s*(?:\((?:₹|rs\.?|inr)\))?\s*(?:as\s+(?:on|of)\s+(?:{_DATE_RX})\s*)?[:\-=]?\s*"
                         + (rf"({_DATE_RX})" if f == "card_due_date" else _AMT_VALUE), re.I)
           for f, lab in _LABELS.items()}
_LABEL_ONLY = {f: re.compile(rf"^\W*(?:{lab})\s*(?:\((?:₹|rs\.?|inr)\))?\s*[:\-=]?\s*$", re.I) for f, lab in _LABELS.items()}
_PERIOD_LABEL = r"(?:statement|billing|bill)\s*(?:period|cycle|date|from|for)?|period|from"
_PERIOD = re.compile(rf"(?:{_PERIOD_LABEL})\s*[:\-]?\s*(?:from\s+)?({_DATE_RX})\s*(?:to|till|until|-)\s*({_DATE_RX})", re.I)
_PERIOD_LABEL_ONLY = re.compile(r"^\W*(?:statement|billing|bill)\s+(?:period|cycle)\s*[:\-]?\s*$", re.I)
_STATEMENT_DATE = re.compile(rf"(?:statement|bill)\s+(?:generation\s+)?date\s*[:\-]?\s*({_DATE_RX})", re.I)


def _value_for(field: str, text: str):
    if field == "card_due_date":
        d = parse_date(text, allow_serial=False)
        return d.isoformat() if d else None
    if parse_date(text, allow_serial=False):
        return None
    m = re.search(_AMT_VALUE, text, re.I)
    return parse_money(m.group(1)) if m else None


def _read_summary(stmt: Statement, lines: list[list[tuple]], extra_text: str = "", period: bool = True) -> None:
    """Fill card / balance summary fields from text lines. Each line is a list of (x0, x1, text) cells.

    Handles "Label: value" on one line, label and value in neighbouring cells, and a row of labels with the
    values in the row below (matched by horizontal position).
    """
    texts = [" ".join(c[2] for c in line) for line in lines] + ([extra_text] if extra_text else [])
    for field, pat in _INLINE.items():
        if getattr(stmt, field) is not None:
            continue
        for t in texts:
            m = pat.search(t)
            if m:
                value = _value_for(field, m.group(1))
                if value is not None:
                    setattr(stmt, field, value)
                    break
    for li, line in enumerate(lines):  # labels whose value sits in the next cell or on the next line
        for ci, (x0, x1, text) in enumerate(line):
            fields = [f for f, pat in _LABEL_ONLY.items() if getattr(stmt, f) is None and pat.match(text)]
            want_period = period and stmt.period_start is None and _PERIOD_LABEL_ONLY.match(text)
            if not fields and not want_period:
                continue
            candidates = line[ci + 1:ci + 2]
            for below in lines[li + 1:li + 3]:
                candidates += [c for c in below if c[0] <= x1 + 2 and c[1] >= x0 - 2] or []
            for c in candidates:
                if want_period:
                    dates = [_date_from_token(m.group(0)) for m in _DATE_TOKEN.finditer(c[2])]
                    dates = [d for d in dates if d]
                    if len(dates) >= 2:
                        stmt.period_start, stmt.period_end = dates[0].isoformat(), dates[1].isoformat()
                        break
                    continue
                value = _value_for(fields[0], c[2])
                if value is not None:
                    setattr(stmt, fields[0], value)
                    break
    if period and stmt.period_start is None:
        for t in texts:
            m = _PERIOD.search(t)
            if m:
                start, end = parse_date(m.group(1), allow_serial=False), parse_date(m.group(2), allow_serial=False)
                if start and end and start <= end:
                    stmt.period_start, stmt.period_end = start.isoformat(), end.isoformat()
                    break
    if period and stmt.period_end is None and _is_card_hint(stmt.source_filename, " ".join(texts)):
        for t in texts:
            m = _STATEMENT_DATE.search(t)
            if m and (d := parse_date(m.group(1), allow_serial=False)):
                stmt.period_end = d.isoformat()
                break


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _pdf_pages(data: bytes) -> list[list[list[tuple]]]:
    """pages -> lines -> words (x0, x1, text, height, y), lines rebuilt from word positions. Scans go through OCR."""
    import fitz  # PyMuPDF

    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            raw = [page.get_text("words") for page in doc]
    except Exception as exc:
        raise StatementError(f"Could not open the PDF: {exc}") from exc
    if any(raw):
        return [_cluster_lines(words) for words in raw]
    from .ocr import ocr_available, ocr_pdf_pages  # scanned statement
    import tempfile

    if not ocr_available():
        raise StatementError("This PDF is a scan and OCR is not installed. Upload the CSV/Excel export instead.")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
    try:
        texts = [text for _, text in ocr_pdf_pages(tmp.name)]
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    return [[[(0.0, float(len(line)), line, 0.0, 12.0 * n)] for n, line in enumerate(l for l in text.splitlines() if l.strip())]
            for text in texts]


def _cluster_lines(words) -> list[list[tuple]]:
    """Group words whose vertical centres line up into one line (table cells come out as separate lines otherwise)."""
    items = sorted(((w[0], w[2], w[4], w[3] - w[1], (w[1] + w[3]) / 2) for w in words if w[4].strip()),
                   key=lambda w: (w[4], w[0]))
    lines: list[list] = []
    centre = None
    for x0, x1, text, h, yc in items:
        if lines and abs(yc - centre) <= max(2.0, 0.45 * h):
            lines[-1].append((x0, x1, text, h, yc))
        else:
            lines.append([(x0, x1, text, h, yc)])
        centre = sum(w[4] for w in lines[-1]) / len(lines[-1])
    return [sorted(line, key=lambda w: w[0]) for line in lines]  # words: (x0, x1, text, height, y-centre)


def _cells(words: list[tuple]) -> list[tuple]:
    """Join neighbouring words into cells; a gap wider than about half a line height starts a new cell."""
    cells: list[list] = []
    for x0, x1, text, h, *_ in words:
        if cells and x0 - cells[-1][1] <= max(4.0, 0.5 * (h or 10)):
            cells[-1][1] = x1
            cells[-1][2] += " " + text
        else:
            cells.append([x0, x1, text])
    return [tuple(c) for c in cells]


_NUMERIC_WORD = re.compile(r"[(+\-]?(?:₹|rs\.?|inr)?[\d,.]*\d[\d,.]*\)?-?", re.I)


def _column_of(word: tuple, layout: list[tuple]) -> int:
    """Column for a word: numbers line up by their right edge (amounts are right-aligned), text by its left edge."""
    x0, x1, text = word[0], word[1], word[2]
    if _NUMERIC_WORD.fullmatch(text):
        return min(range(len(layout)), key=lambda k: abs(layout[k][1] - x1))
    return max((k for k in range(len(layout)) if layout[k][0] <= x0 + 3), default=0)


def _pdf_header_candidates(pages: list[list[list[tuple]]]) -> list[list[str]]:
    """Distinct lines that look like a table header (a summary row of labels can look like one too)."""
    seen, out = set(), []
    for lines in pages:
        for words in lines:
            cells = _cells(words)
            if len(cells) >= 3 and _is_header([c[2] for c in cells]):
                key = tuple(_norm(c[2]) for c in cells)
                if key not in seen:
                    seen.add(key)
                    out.append([c[2] for c in cells])
    return out[:6]


def _pdf_table(pages: list[list[list[tuple]]], filename: str, canon: list[str]) -> tuple[list[list], list]:
    """Rows rebuilt by column position under the header ``canon`` (repeated on later pages after page breaks)."""
    canon_key = tuple(_norm(c) for c in canon)
    rows: list[list] = []
    sources: list = []
    layout: Optional[list[tuple]] = None
    for p, lines in enumerate(pages, 1):
        for ln, words in enumerate(lines, 1):
            cells = _cells(words)
            if tuple(_norm(c[2]) for c in cells) == canon_key:
                if layout is None:
                    rows.append(list(canon))
                    sources.append(f"{filename} page {p}, line {ln}")
                layout = cells
                rows.append([""] * len(canon))  # header seen again: break any wrapped-description joining
                sources.append(None)
                continue
            if layout is None:
                continue
            row = [""] * len(canon)
            for word in words:
                k = _column_of(word, layout)
                row[k] = (row[k] + " " + word[2]).strip()
            if ln > 1 and words[0][4] - lines[ln - 2][0][4] > 1.9 * max(words[0][3], 1.0):
                rows.append([""] * len(canon))  # a visible gap above this line: not a wrapped description
                sources.append(None)
            rows.append(row)
            sources.append(f"{filename} page {p}, line {ln}")
        if layout is not None:
            rows.append([""] * len(canon))  # page break
            sources.append(None)
    return rows, sources


_AMT_TOKEN = (r"\(?[+-]?\s?(?:₹|rs\.?|inr)?\s?[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2}\)?(?:\s?(?:cr|dr)\b\.?)?-?"
              r"|(?:₹|rs\.?|inr)\s?\d{1,3}(?:,\d{2,3})*(?:\s?(?:cr|dr)\b\.?)?"
              r"|\d{1,3}(?:,\d{2,3})+(?:\s?(?:cr|dr)\b\.?)?")
_TAIL_AMOUNT = re.compile(rf"(?:^|\s)({_AMT_TOKEN})\s*$", re.I)
_LINE_START = re.compile(rf"^\s*({_DATE_RX}){_TIME}\s+(.+)$", re.I)
_LEAD_DATE = re.compile(rf"^\s*(?:{_DATE_RX}){_TIME}\s+", re.I)


def _pdf_lines_statement(pages: list[list[list[tuple]]], filename: str, card: bool, opening: Optional[float]) -> Statement:
    """Fallback for PDFs without a recognisable header row: 'date  description  amount [Cr/Dr] [balance]' lines."""
    stmt = Statement(source_filename=filename)
    pending: list[dict] = []
    prev_balance = opening
    for p, lines in enumerate(pages, 1):
        for ln, words in enumerate(lines, 1):
            text = " ".join(w[2] for w in words)
            m = _LINE_START.match(text)
            if not m:
                continue
            d = parse_date(m.group(1), allow_serial=False)
            if d is None:
                continue
            rest, amounts = m.group(2), []
            while len(amounts) < 3 and (t := _TAIL_AMOUNT.search(rest)):
                amounts.insert(0, t.group(1))
                rest = rest[:t.start()].rstrip()
            desc = _LEAD_DATE.sub("", rest).strip()  # drop a second (value) date
            if not amounts or not re.search(r"[A-Za-z]", desc):
                continue
            parsed = [_money(a) for a in amounts]
            balance = None
            if len(parsed) == 1:
                (amount, marker), = parsed
            elif len(parsed) == 2:
                (amount, marker), (balance, _) = parsed
            else:  # debit, credit, balance
                (dr, _), (cr, _), (balance, _) = parsed
                amount, marker = (dr, "dr") if dr else (cr, "cr")
            if not amount:
                continue
            txn_type = {"cr": "credit", "dr": "debit"}.get(marker)
            if txn_type is None and marker == "neg":
                txn_type = "credit" if card else "debit"
            if txn_type is None and balance is not None and prev_balance is not None:
                txn_type = "credit" if balance > prev_balance else "debit"
            if txn_type is None:
                txn_type = "credit" if (_credit_like(desc) if card else _guess_type(desc) == "credit") else "debit"
            pending.append({"date": d, "desc": desc, "amount": abs(amount), "type": txn_type, "balance": balance,
                            "source": f"{filename} page {p}, line {ln}"})
            if balance is not None:
                prev_balance = balance
    for e in pending:
        _add(stmt, e["date"], e["desc"], e["amount"], e["type"], e["balance"], e["source"])
    return stmt


def _parse_pdf(data: bytes, filename: str) -> Statement:
    pages = _pdf_pages(data)
    all_lines = [[(w[0], w[1], w[2]) for w in _cells(line)] for lines in pages for line in lines]
    full = "\n".join(" ".join(c[2] for c in line) for line in all_lines)
    card = _is_card_hint(filename, full)
    summary = Statement(source_filename=filename)
    _read_summary(summary, all_lines)

    candidates: list[Statement] = []
    for canon in _pdf_header_candidates(pages):
        rows, sources = _pdf_table(pages, filename, canon)
        try:
            candidates.append(_parse_table(rows, filename, sources=sources, meta_lines=[], card_hint=card))
        except StatementError:
            pass
    candidates.append(_pdf_lines_statement(pages, filename, card, summary.opening_balance))
    stmt = max(candidates, key=lambda s: len(s.transactions))  # tables come first, so they win a tie
    if not stmt.transactions:
        raise StatementError("No transactions could be read from this PDF. Try the CSV or Excel export from your bank.")
    for field in ("opening_balance", "closing_balance", "card_total_due", "card_minimum_due", "card_due_date",
                  "card_credit_limit", "period_start", "period_end"):
        if getattr(summary, field) is not None:
            setattr(stmt, field, getattr(summary, field))
    if card or stmt.card_due_date or stmt.card_minimum_due:
        stmt.kind = "card"
    return stmt


# ---------------------------------------------------------------------------
# Spreadsheet readers
# ---------------------------------------------------------------------------
def _rows_from_frame(frame) -> list[list]:
    def clean(v):
        if v is None:
            return ""
        try:
            import pandas as pd

            if pd.isna(v):
                return ""
        except (TypeError, ValueError):
            pass
        return v

    return [[clean(v) for v in row] for row in frame.itertuples(index=False)]


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:2048]:
        raise InvalidFileError("This is not a text file.")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise InvalidFileError("Could not read the text in this file.")


def _parse_delimited(text: str, filename: str) -> Statement:
    """Try every likely delimiter and keep the reading that finds the most transactions."""
    import csv

    best, error = None, None
    for delim in [d for d in (",", ";", "\t", "|") if d in text] or [","]:
        try:
            rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
            stmt = _parse_table(rows, filename)
        except (csv.Error, StatementError) as exc:
            error = exc
            continue
        if best is None or len(stmt.transactions) > len(best.transactions):
            best = stmt
    if best is None:
        if isinstance(error, StatementError):
            raise error
        raise StatementError(f"Could not read the file: {error}")
    return best


def _parse_workbook(sheets: dict[str, list[list]], filename: str) -> Statement:
    """Every sheet is tried; the one with the most transactions is used, summary values come from all sheets."""
    best, best_name, error = None, None, None
    multi = sum(1 for rows in sheets.values() if rows) > 1
    lines_by_sheet = {name: _grid_lines(rows[:300]) for name, rows in sheets.items()}
    hint_text = " ".join(" ".join(c[2] for c in line) for lines in lines_by_sheet.values() for line in lines[:100])
    for name, rows in sheets.items():
        if not rows:
            continue
        prefix = f"{filename} [{name}]" if multi else filename
        try:
            stmt = _parse_table(rows, filename, sources=[f"{prefix} row {i + 1}" for i in range(len(rows))],
                                hint_text=hint_text)
        except StatementError as exc:
            error = exc
            continue
        if best is None or len(stmt.transactions) > len(best.transactions):
            best, best_name = stmt, name
    if best is None:
        raise error or StatementError("This workbook has no sheets with transactions.")
    others = [line for name, lines in lines_by_sheet.items() if name != best_name for line in lines]
    _read_summary(best, others, period=best.kind == "card")  # e.g. a separate "Summary" sheet
    return best


def _html_tables(text: str) -> dict[str, list[list]]:
    """Tables from an HTML page (some banks' ".xls" downloads are really HTML)."""
    from html.parser import HTMLParser

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows, self.row, self.cell = [], None, None

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row = []
            elif tag in ("td", "th"):
                self.cell = []
            elif tag == "br" and self.cell is not None:
                self.cell.append(" ")

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.cell is not None and self.row is not None:
                self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip())
                self.cell = None
            elif tag == "tr" and self.row is not None:
                if any(self.row):
                    self.rows.append(self.row)
                self.row = None

        def handle_data(self, data):
            if self.cell is not None:
                self.cell.append(data)

    p = P()
    p.feed(text)
    return {"Sheet1": p.rows}


def _spreadsheet_xml(text: str) -> dict[str, list[list]]:
    """Excel 2003 XML ("SpreadsheetML") saved with a .xls name."""
    import xml.etree.ElementTree as ET

    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    root = ET.fromstring(text)
    sheets = {}
    for ws in root.findall("ss:Worksheet", ns):
        rows = []
        for r in ws.findall(".//ss:Row", ns):
            row = []
            for c in r.findall("ss:Cell", ns):
                idx = c.get(f"{{{ns['ss']}}}Index")
                if idx:
                    row += [""] * (int(idx) - 1 - len(row))
                d = c.find("ss:Data", ns)
                row.append("".join(d.itertext()).strip() if d is not None else "")
            rows.append(row)
        sheets[ws.get(f"{{{ns['ss']}}}Name") or f"Sheet{len(sheets) + 1}"] = rows
    return sheets


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------
def _add(stmt: Statement, d: date, desc_raw: str, amount: float, txn_type: str, balance: Optional[float], source: str) -> None:
    if amount <= 0 or len(stmt.transactions) >= MAX_TRANSACTIONS:
        return
    description = mask_description(desc_raw)
    category, brand = categorise(description, txn_type)
    if category == "Fees & charges":  # name the fee itself, not a vendor
        label = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z ]", " ", description)).strip()[:40].capitalize() or "Bank charge"
        merchant, key = label, label.lower()
    else:
        merchant, key = merchant_of(description, brand)
    stmt.transactions.append(Transaction(
        txn_id=f"T{len(stmt.transactions) + 1}", date=d.isoformat(), description=description, merchant=merchant,
        merchant_key=key, amount=round(amount, 2), type=txn_type, category=category,
        fee=(category == "Fees & charges"), balance=balance, source=source))


def _finish(stmt: Statement) -> Statement:
    if not stmt.transactions:
        raise StatementError("No transactions found in this file.")
    stmt.transactions.sort(key=lambda t: (t.date, int(t.txn_id[1:])))
    stmt.period_start = stmt.period_start or stmt.transactions[0].date
    stmt.period_end = stmt.period_end or stmt.transactions[-1].date
    if stmt.card_due_date:
        stmt.kind = "card"
    for t in stmt.transactions:  # card bills are compared bill-to-bill, bank statements month-to-month
        t.period = t.period or (stmt.period_end[:7] if stmt.kind == "card" else t.date[:7])
    if stmt.closing_balance is None:
        with_bal = [t for t in stmt.transactions if t.balance is not None]
        if with_bal:
            stmt.closing_balance = with_bal[-1].balance
    if len(stmt.transactions) >= MAX_TRANSACTIONS:
        stmt.warnings.append(f"Only the first {MAX_TRANSACTIONS} transactions were read.")
    return stmt


def parse_statement(filename: str, data: bytes) -> Statement:
    """Read a CSV / TSV / TXT, Excel (.xlsx, or HTML/XML .xls exports) or PDF statement into masked transactions."""
    name = Path(filename or "statement").name
    ext = Path(name).suffix.lower()
    if not data:
        raise InvalidFileError("The file is empty.")
    if len(data) > MAX_STATEMENT_BYTES:
        raise InvalidFileError("The statement is too large (limit 10 MB).")
    if ext in _MAGIC and not data.startswith(_MAGIC[ext]):
        raise InvalidFileError(f"This is not a real {ext[1:].upper()} file.")
    if ext == ".pdf":
        return _finish(_parse_pdf(data, name))
    if ext in (".csv", ".tsv", ".txt"):
        return _finish(_parse_delimited(_decode_text(data), name))
    if ext == ".xlsx":
        import pandas as pd

        try:
            frames = pd.read_excel(io.BytesIO(data), header=None, dtype=object, engine="openpyxl", sheet_name=None)
        except Exception as exc:
            raise StatementError(f"Could not read the Excel file: {exc}") from exc
        return _finish(_parse_workbook({n: _rows_from_frame(f) for n, f in frames.items()}, name))
    if ext == ".xls":
        if data.startswith(b"\xd0\xcf\x11\xe0"):
            raise InvalidFileError("This is the old Excel 97-2003 (.xls) format. Open it in Excel or Google Sheets "
                                   "and save it as .xlsx or CSV, then upload again.")
        text = _decode_text(data)
        head = text.lstrip()[:2000].lower()
        try:
            if "urn:schemas-microsoft-com:office:spreadsheet" in head:
                sheets = _spreadsheet_xml(text)
            elif "<table" in text.lower() or head.startswith(("<html", "<!doctype", "<meta")):
                sheets = _html_tables(text)
            else:
                return _finish(_parse_delimited(text, name))  # some banks send tab-separated text as .xls
        except StatementError:
            raise
        except Exception as exc:
            raise StatementError(f"Could not read the .xls file: {exc}") from exc
        return _finish(_parse_workbook(sheets, name))
    raise InvalidFileError("Upload a CSV, Excel (.xlsx) or PDF statement.")


def merge_statements(statements: list[Statement]) -> Statement:
    """Combine several statements (e.g. two months), dropping rows that appear in both."""
    if len(statements) == 1:
        return statements[0]
    merged = Statement(source_filename=", ".join(s.source_filename for s in statements))
    seen = set()
    for s in sorted(statements, key=lambda s: s.period_start or ""):
        for t in s.transactions:
            key = (t.date, t.amount, t.type, t.description)
            if key in seen:
                continue
            seen.add(key)
            merged.transactions.append(t.model_copy(update={"txn_id": f"T{len(merged.transactions) + 1}"}))
        for field in ("card_total_due", "card_minimum_due", "card_due_date", "card_credit_limit", "closing_balance"):
            if getattr(s, field) is not None:
                setattr(merged, field, getattr(s, field))  # latest statement wins
        if s.kind == "card":
            merged.kind = "card"
        merged.warnings += s.warnings
    merged.opening_balance = statements[0].opening_balance
    merged.period_start = min((s.period_start for s in statements if s.period_start), default=None)
    merged.period_end = max((s.period_end for s in statements if s.period_end), default=None)
    return _finish(merged)
